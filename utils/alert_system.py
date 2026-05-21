import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import queue
import threading


class AlertSystem:
    # Severity Levels
    LEVEL_INFO = "INFO"
    LEVEL_WARNING = "WARNING"
    LEVEL_CRITICAL = "CRITICAL"

    # Strict Alert Definitions
    ALERT_NO_PERSON = "NO_PERSON"
    ALERT_SINGLE_PERSON = "SINGLE_PERSON"
    ALERT_UNAUTHORIZED = "UNAUTHORIZED_ACCESS"
    ALERT_EXCESS_PEOPLE = "EXCESS_PEOPLE"
    ALERT_AUTH_SUCCESS = "AUTHORIZED_ACCESS"
    ALERT_SPOOFING_DETECTED = "SPOOFING_DETECTED"

    def __init__(self, config: dict):
        self.config = config
        self._setup_logging()
        
        # Priority Cooldown Map (Critical errors bypass throttling)
        self.cooldown_map = {
            self.ALERT_AUTH_SUCCESS: config.get('cooldown_success', 5),
            self.ALERT_NO_PERSON: config.get('cooldown_idle', 30),
            self.ALERT_SINGLE_PERSON: config.get('cooldown_idle', 10),
            self.ALERT_EXCESS_PEOPLE: 0,       # Zero cooldown: process instantly
            self.ALERT_UNAUTHORIZED: 0,         # Zero cooldown
            self.ALERT_SPOOFING_DETECTED: 0     # Zero cooldown
        }
        self.last_alerts: Dict[str, float] = {}
        
        # Thread-safe Task Queue for Network Operations
        self.alert_queue: queue.Queue = queue.Queue(maxsize=100)
        self.running = True
        
        # Local Flag Actions
        self.sound_enabled = config.get('enable_sound', True)
        self.hardware_relay_enabled = config.get('enable_hardware_relay', False)

        # Initialize Background Dispatcher Thread
        self.dispatcher_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.dispatcher_thread.start()
        
        self.logger.info("Asynchronous Alert Engine active and listening.")

    def _setup_logging(self):
        log_file = self.config.get('log_file', 'logs/surveillance.log')
        log_level = self.config.get('level', 'INFO')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s [%(levelname)s] [THREAD:%(threadName)s] %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _can_send_alert(self, alert_type: str) -> bool:
        current_time = time.time()
        cooldown = self.cooldown_map.get(alert_type, 10)
        
        if cooldown == 0:
            return True
            
        last_time = self.last_alerts.get(alert_type, 0.0)
        if current_time - last_time >= cooldown:
            self.last_alerts[alert_type] = current_time
            return True
        return False

    def send_alert(self, alert_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Non-blocking ingestion point for edge inferences."""
        if not self._can_send_alert(alert_type):
            return

        alert_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': alert_type,
            'message': message,
            'data': data or {}
        }

        # 1. Inline Immediate Local Operations (Microsecond latency)
        self._log_alert(alert_type, message, alert_data)
        
        if self.sound_enabled and alert_type in [self.ALERT_UNAUTHORIZED, self.ALERT_SPOOFING_DETECTED]:
            self._trigger_local_buzzer()
            
        if self.hardware_relay_enabled and alert_type == self.ALERT_SPOOFING_DETECTED:
            self._trigger_lockdown_relay()

        # 2. Push heavy network distributions to background worker queue
        try:
            self.alert_queue.put_nowait(alert_data)
        except queue.Full:
            self.logger.critical("Alert drops occurring! Internal buffer full.")

    def _log_alert(self, alert_type: str, message: str, alert_data: dict):
        if alert_type == self.ALERT_AUTH_SUCCESS:
            self.logger.info(f"[{alert_type}] {message}")
        elif alert_type in [self.ALERT_SINGLE_PERSON, self.ALERT_NO_PERSON]:
            self.logger.warning(f"[{alert_type}] {message}")
        else:
            self.logger.error(f"🚨 CRITICAL EVENT: [{alert_type}] {message}")

    def _trigger_local_buzzer(self):
        # Local system call - clean execution context
        print('\a', end='', flush=True) 

    def _trigger_lockdown_relay(self):
        """Simulate tripping an edge GPIO or network relay switch to lock local barriers."""
        self.logger.critical("SYSTEM ACTION: TRIP FAULT RELAY -> VAULT PORTAL IMMOBILIZED")

    def _worker_loop(self):
        """Continuous background thread processing slow I/O out-of-band."""
        # Lazily import heavy libraries here to save main execution bootstrap overhead
        import requests 
        
        while self.running:
            try:
                # Blocks worker thread, never the frame processing frame loop
                alert_data = self.alert_queue.get(timeout=1.0)
                
                # Handle downstream Webhooks / Integrations securely
                if self.config.get('enable_webhook', False):
                    self._dispatch_webhook(alert_data, requests)
                    
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in alert worker routine: {e}", exc_info=True)

    def _dispatch_webhook(self, alert_data: dict, network_lib):
        wh_config = self.config.get('webhook', {})
        url = wh_config.get('url')
        if not url:
            return
            
        try:
            response = network_lib.post(
                url, 
                json=alert_data, 
                headers=wh_config.get('headers', {}), 
                timeout=3.0
            )
            if response.status_code != 200:
                self.logger.warning(f"External notification gateway down. Status: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Asynchronous webhook dispatch dropped: {e}")

    def log_event(self, event_type: str, message: str, level: str = "INFO"):
        """Convenience method for general event logging."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"[{event_type}] {message}")

    def shutdown(self):
        """Gracefully drain queues during maintenance or software updates."""
        self.running = False
        if self.dispatcher_thread.is_alive():
            self.dispatcher_thread.join(timeout=5.0)

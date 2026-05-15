import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import json

class AlertSystem:
    ALERT_NO_PERSON = "NO_PERSON"
    ALERT_SINGLE_PERSON = "SINGLE_PERSON"
    ALERT_UNAUTHORIZED = "UNAUTHORIZED_ACCESS"
    ALERT_EXCESS_PEOPLE = "EXCESS_PEOPLE"
    ALERT_AUTH_SUCCESS = "AUTHORIZED_ACCESS"
    
    def __init__(self, config: dict):
        self.config = config
        self.cooldown = config.get('cooldown_seconds', 10)
        self.last_alerts = {}  # {alert_type: timestamp}
        
        self._setup_logging()
        
        self.sound_enabled = config.get('enable_sound', True)
        
        self.email_enabled = config.get('enable_email', False)
        if self.email_enabled:
            self._setup_email()
        
        self.webhook_enabled = config.get('enable_webhook', False)
        if self.webhook_enabled:
            self.webhook_url = config.get('webhook', {}).get('url')
            self.webhook_headers = config.get('webhook', {}).get('headers', {})
        
        print("✓ Alert system initialized")
    
    def _setup_logging(self):
        """Setup file logging"""
        log_file = self.config.get('log_file', 'logs/surveillance.log')
        log_level = self.config.get('level', 'INFO')
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_email(self):
        email_config = self.config.get('email', {})
        self.smtp_server = email_config.get('smtp_server')
        self.smtp_port = email_config.get('smtp_port')
        self.sender_email = email_config.get('sender_email')
        self.sender_password = email_config.get('sender_password')
        self.recipient_emails = email_config.get('recipient_emails', [])
    
    def _can_send_alert(self, alert_type: str) -> bool:
        current_time = time.time()
        last_time = self.last_alerts.get(alert_type, 0)
        
        if current_time - last_time >= self.cooldown:
            self.last_alerts[alert_type] = current_time
            return True
        return False
    
    def send_alert(self, alert_type: str, message: str, data: Optional[Dict] = None):
        if not self._can_send_alert(alert_type):
            return
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'data': data or {}
        }
        
        self._log_alert(alert_type, message, alert_data)
        
        if self.sound_enabled and alert_type != self.ALERT_AUTH_SUCCESS:
            self._play_sound_alert(alert_type)
        
        if self.email_enabled:
            self._send_email_alert(alert_data)
        
        if self.webhook_enabled:
            self._send_webhook_alert(alert_data)
    
    def _log_alert(self, alert_type: str, message: str, alert_data: dict):
        if alert_type == self.ALERT_AUTH_SUCCESS:
            self.logger.info(message)
        elif alert_type in [self.ALERT_SINGLE_PERSON, self.ALERT_NO_PERSON]:
            self.logger.warning(message)
        else:
            self.logger.error(message)
        
        self.logger.debug(f"Alert data: {json.dumps(alert_data, indent=2)}")
    
    def _play_sound_alert(self, alert_type: str):
        try:
            print('\a')            
        except Exception as e:
            self.logger.debug(f"Sound alert failed: {e}")
    
    def _send_email_alert(self, alert_data: dict):
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = f"[SURVEILLANCE ALERT] {alert_data['type']}"
            
            body = f"""
            Surveillance Alert
            
            Time: {alert_data['timestamp']}
            Type: {alert_data['type']}
            Message: {alert_data['message']}
            
            Details:
            {json.dumps(alert_data['data'], indent=2)}
            
            --
            Strongroom Surveillance System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.logger.info("Email alert sent")
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
    
    def _send_webhook_alert(self, alert_data: dict):
        try:
            import requests
            
            response = requests.post(
                self.webhook_url,
                json=alert_data,
                headers=self.webhook_headers,
                timeout=5
            )
            
            if response.status_code == 200:
                self.logger.info("Webhook alert sent")
            else:
                self.logger.warning(f"Webhook returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
    
    def log_event(self, event_type: str, message: str, level: str = "INFO"):
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"[{event_type}] {message}")

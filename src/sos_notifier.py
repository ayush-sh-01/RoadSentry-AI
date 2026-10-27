"""
RoadSentry AI - Accident SOS Notifier Module
Dispatches critical accident alerts to configured emergency email addresses and mobile numbers,
with real delivery status tracking and graceful offline/simulation handling.
"""

from typing import Dict, Any, Optional, List
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class SOSNotifier:
    """
    Emergency SOS Notification Engine.
    Handles automated email/SMS dispatch when road incidents or collisions are detected.
    """

    def __init__(
        self,
        recipient_email: str = "traffic_control@emergency.gov",
        recipient_phone: str = "+91-9876543210",
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        cooldown_seconds: int = 45,
        enabled: bool = True,
        **kwargs
    ):
        self.recipient_email = recipient_email
        self.recipient_phone = recipient_phone
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled

        self.last_dispatch_time: float = 0.0
        self.dispatch_history: List[Dict[str, Any]] = []

        self.last_status: Dict[str, Any] = {
            "status": "STANDBY",
            "delivered": False,
            "recipient_email": self.recipient_email,
            "recipient_phone": self.recipient_phone,
            "message": "SOS Dispatcher ready on standby. Monitoring for collision events.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def update_config(
        self,
        recipient_email: Optional[str] = None,
        recipient_phone: Optional[str] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None
    ):
        """Update notification credentials dynamically from UI controls."""
        if recipient_email is not None and recipient_email.strip():
            self.recipient_email = recipient_email.strip()
        if recipient_phone is not None and recipient_phone.strip():
            self.recipient_phone = recipient_phone.strip()
        if smtp_user is not None:
            self.smtp_user = smtp_user.strip()
        if smtp_password is not None:
            self.smtp_password = smtp_password.strip()
        if smtp_server is not None and smtp_server.strip():
            self.smtp_server = smtp_server.strip()
        if smtp_port is not None:
            self.smtp_port = smtp_port

        self.last_status["recipient_email"] = self.recipient_email
        self.last_status["recipient_phone"] = self.recipient_phone

    def send_sos(
        self,
        incident: Dict[str, Any],
        location_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispatches an SOS alert for a detected road incident.
        Respects cooldown timer to avoid flooding during continuous collision frames.
        """
        if not self.enabled:
            return self.last_status

        curr_time = time.time()
        if curr_time - self.last_dispatch_time < self.cooldown_seconds:
            # Within cooldown
            return self.last_status

        self.last_dispatch_time = curr_time
        timestamp = incident.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        inc_id = incident.get("incident_id", f"INC-{int(curr_time)}")
        severity = incident.get("severity", "CRITICAL")
        v_ids = incident.get("vehicle_ids", [])
        v_classes = incident.get("classes", ["vehicle"])

        loc = location_data or {
            "junction_name": "Monitored Road Junction",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "is_simulated_gps": True
        }

        gps_label = "Simulated CCTV GPS Anchor" if loc.get("is_simulated_gps") else "Live Hardware GPS"
        loc_str = f"{loc.get('junction_name', 'Unknown')} (Lat: {loc.get('latitude', 0.0)}, Lon: {loc.get('longitude', 0.0)} [{gps_label}])"

        subject = f"🚨 [RoadSentry AI] CRITICAL ACCIDENT SOS: {loc.get('junction_name', 'Junction')}"

        plain_text = f"""
=====================================================
🚨 ROADSENTRY AI - CRITICAL ROAD INCIDENT SOS ALERT
=====================================================

Incident ID   : {inc_id}
Severity      : {severity}
Timestamp     : {timestamp}
Location      : {loc_str}
Vehicles      : #{v_ids} (Classes: {v_classes})
Collision IoU : {incident.get('iou', 'N/A')}

ACTION REQUIRED:
Automated vision anomaly detection flagged a critical vehicle collision.
Please dispatch nearest highway emergency patrol and medical units immediately.

Sent via RoadSentry AI Autonomous Traffic Safety System.
=====================================================
"""

        delivery_result = self._dispatch_email(
            subject=subject,
            body=plain_text,
            to_email=self.recipient_email
        )

        status_record = {
            "incident_id": inc_id,
            "timestamp": timestamp,
            "severity": severity,
            "location": loc_str,
            "recipient_email": self.recipient_email,
            "recipient_phone": self.recipient_phone,
            "status": delivery_result["status"],
            "delivered": delivery_result["delivered"],
            "message": delivery_result["message"]
        }

        self.last_status = status_record
        self.dispatch_history.insert(0, status_record)
        return status_record

    def _dispatch_email(self, subject: str, body: str, to_email: str) -> Dict[str, Any]:
        """Handles physical SMTP transport with graceful fallback."""
        if not to_email:
            return {
                "status": "FAILED_NO_EMAIL",
                "delivered": False,
                "message": "No recipient email configured. Please enter email in settings or sidebar."
            }

        # If sender credentials are provided, attempt real SMTP transmission
        if self.smtp_user and self.smtp_password:
            try:
                msg = MIMEMultipart()
                msg["From"] = self.smtp_user
                msg["To"] = to_email
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))

                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=8) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)

                return {
                    "status": "DELIVERED",
                    "delivered": True,
                    "message": f"✅ Email successfully delivered to {to_email} via {self.smtp_user}."
                }
            except smtplib.SMTPAuthenticationError:
                return {
                    "status": "FAILED_AUTH",
                    "delivered": False,
                    "message": "❌ Gmail Authentication Failed. If using Gmail, make sure you use a 16-character 'Google App Password' (not your personal account password)."
                }
            except Exception as e:
                return {
                    "status": "FAILED_SMTP_ERROR",
                    "delivered": False,
                    "message": f"⚠️ Delivery Failed: {str(e)}. Check internet connection or SMTP credentials."
                }
        else:
            # Simulation / Demo Mode (No Sender SMTP credentials configured yet)
            return {
                "status": "SIMULATED_ONLY",
                "delivered": False,
                "message": f"⚠️ Email NOT sent to inbox: Sender Gmail and App Password not provided. Enter sender credentials in the sidebar to send real emails to {to_email}."
            }

    def test_dispatch(self, test_email: Optional[str] = None) -> Dict[str, Any]:
        """Manually triggers a test alert to verify recipient configuration."""
        target = (test_email or self.recipient_email).strip()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        subject = "🧪 [RoadSentry AI] TEST SOS Notification"
        body = f"""
This is a test notification from RoadSentry AI.
Timestamp: {timestamp}
Status: System operational.
Recipient verified: {target}
"""
        res = self._dispatch_email(subject, body, target)
        record = {
            "incident_id": f"TEST-{int(time.time())}",
            "timestamp": timestamp,
            "severity": "TEST",
            "location": "Local Node Test",
            "recipient_email": target,
            "recipient_phone": self.recipient_phone,
            "status": res["status"],
            "delivered": res["delivered"],
            "message": res["message"]
        }
        self.last_status = record
        self.dispatch_history.insert(0, record)
        return record

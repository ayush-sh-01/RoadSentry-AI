"""
RoadSentry AI - Accident SOS Notifier Module
Dispatches critical accident alerts to configured emergency email addresses and mobile numbers,
with real delivery status tracking and graceful offline/simulation handling.
"""

from typing import Dict, Any, Optional, List
import time
import smtplib
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import requests
except ImportError:
    requests = None


class SOSNotifier:
    """
    Emergency SOS Notification Engine.
    Handles automated multi-channel dispatch (Email, Free Instant Mobile Push, Webhooks)
    when road incidents or collisions are detected.
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
        webhook_url: str = "",
        callmebot_apikey: str = "",
        ntfy_topic: str = "",
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
        self.webhook_url = webhook_url
        self.callmebot_apikey = callmebot_apikey
        self.fast2sms_apikey = kwargs.get("fast2sms_apikey", "")
        self.twilio_account_sid = kwargs.get("twilio_account_sid", "")
        self.twilio_auth_token = kwargs.get("twilio_auth_token", "")
        self.twilio_from_number = kwargs.get("twilio_from_number", "")
        
        # Auto-generate a clean phone-based topic for zero-setup instant phone notifications
        cleaned_phone = "".join(filter(str.isdigit, recipient_phone))
        self.ntfy_topic = ntfy_topic.strip() if ntfy_topic else f"roadsentry_{cleaned_phone or 'alert'}"

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
        smtp_port: Optional[int] = None,
        webhook_url: Optional[str] = None,
        callmebot_apikey: Optional[str] = None,
        ntfy_topic: Optional[str] = None
    ):
        """Update notification credentials dynamically from UI controls."""
        if recipient_email is not None and recipient_email.strip():
            self.recipient_email = recipient_email.strip()
        if recipient_phone is not None and recipient_phone.strip():
            self.recipient_phone = recipient_phone.strip()
            cleaned = "".join(filter(str.isdigit, self.recipient_phone))
            if not self.ntfy_topic or self.ntfy_topic.startswith("roadsentry_"):
                self.ntfy_topic = f"roadsentry_{cleaned or 'alert'}"
        if smtp_user is not None:
            self.smtp_user = smtp_user.strip()
        if smtp_password is not None:
            self.smtp_password = smtp_password.strip()
        if smtp_server is not None and smtp_server.strip():
            self.smtp_server = smtp_server.strip()
        if smtp_port is not None:
            self.smtp_port = smtp_port
        if webhook_url is not None:
            self.webhook_url = webhook_url.strip()
        if callmebot_apikey is not None:
            self.callmebot_apikey = callmebot_apikey.strip()
        if ntfy_topic is not None and ntfy_topic.strip():
            self.ntfy_topic = ntfy_topic.strip()

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

        # 1. Attempt Real Mobile SMS (Fast2SMS or Twilio)
        sms_result = self._dispatch_sms(
            phone_number=self.recipient_phone,
            message=f"CRITICAL ACCIDENT ALERT!\nID: {inc_id}\nLoc: {loc.get('junction_name', 'Road')}\nVehicles: #{v_ids}\nAction: Emergency response needed."
        )

        # 2. Attempt Instant Mobile Phone Push Notification (direct to device screen)
        phone_push_res = self._dispatch_phone_push(
            title=f"🚨 ACCIDENT DETECTED: {loc.get('junction_name', 'Junction')}",
            body=f"Severity: {severity}\nLocation: {loc_str}\nVehicles: #{v_ids} ({v_classes})"
        )

        # 3. Attempt Email Dispatch (if configured)
        delivery_result = self._dispatch_email(
            subject=subject,
            body=plain_text,
            to_email=self.recipient_email
        )

        # 4. Attempt Webhook Dispatch (Discord / Telegram / Custom webhook)
        webhook_res = self._dispatch_webhook(
            incident=incident,
            location_str=loc_str,
            subject=subject,
            body=plain_text
        )

        # Combine delivery status
        is_delivered = sms_result.get("delivered", False) or phone_push_res["delivered"] or delivery_result["delivered"] or webhook_res.get("delivered", False)
        
        messages = []
        if sms_result.get("delivered"):
            messages.append(f"📱 {sms_result['message']}")
        elif sms_result.get("status") not in ("SKIPPED", "NO_KEY"):
            messages.append(f"📱 SMS: {sms_result.get('message')}")

        if phone_push_res["delivered"]:
            messages.append(f"📲 {phone_push_res['message']}")
        if delivery_result["delivered"]:
            messages.append(delivery_result["message"])
        if webhook_res.get("delivered"):
            messages.append(f"🌐 {webhook_res['message']}")

        if not messages:
            messages.append(sms_result.get("message") or delivery_result["message"])

        final_status = "DELIVERED" if is_delivered else (delivery_result["status"] if delivery_result.get("status") == "SIMULATED_ONLY" else (sms_result.get("status") or delivery_result["status"]))

        status_record = {
            "incident_id": inc_id,
            "timestamp": timestamp,
            "severity": severity,
            "location": loc_str,
            "recipient_email": self.recipient_email,
            "recipient_phone": self.recipient_phone,
            "status": final_status,
            "delivered": is_delivered,
            "message": " | ".join(messages),
            "sms": sms_result,
            "phone_push": phone_push_res,
            "webhook": webhook_res
        }

        self.last_status = status_record
        self.dispatch_history.insert(0, status_record)
        return status_record

    def _dispatch_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """
        Dispatches a direct SMS to the recipient's phone number.
        Supports Fast2SMS (Indian numbers) and Twilio (Global numbers).
        """
        cleaned_digits = "".join(filter(str.isdigit, phone_number))
        if not cleaned_digits:
            return {"delivered": False, "status": "FAILED_NO_NUMBER", "message": "No mobile number provided."}

        # Fast2SMS handler for Indian numbers (10 digits)
        if self.fast2sms_apikey:
            try:
                # Fast2SMS requires 10-digit format
                num_10 = cleaned_digits[-10:]
                url = "https://www.fast2sms.com/dev/bulkV2"
                params = {
                    "authorization": self.fast2sms_apikey,
                    "route": "q",
                    "message": message[:160],
                    "language": "english",
                    "flash": 0,
                    "numbers": num_10
                }
                headers = {"Cache-Control": "no-cache"}
                resp = requests.get(url, params=params, headers=headers, timeout=6)
                data = resp.json()
                if data.get("return") is True:
                    return {
                        "delivered": True,
                        "status": "DELIVERED",
                        "message": f"Real SMS delivered to {num_10} via Fast2SMS gateway."
                    }
                else:
                    err_msg = data.get("message", ["SMS Gateway error"])[0] if isinstance(data.get("message"), list) else data.get("message", "Error")
                    return {
                        "delivered": False,
                        "status": "FAILED_GATEWAY",
                        "message": f"Fast2SMS rejected: {err_msg}"
                    }
            except Exception as e:
                return {"delivered": False, "status": "ERROR", "message": f"SMS connection error: {str(e)}"}

        # Twilio SMS Handler
        if self.twilio_account_sid and self.twilio_auth_token and self.twilio_from_number:
            try:
                from twilio.rest import Client
                client = Client(self.twilio_account_sid, self.twilio_auth_token)
                target_num = f"+{cleaned_digits}" if not phone_number.startswith("+") else phone_number
                msg = client.messages.create(
                    body=message,
                    from_=self.twilio_from_number,
                    to=target_num
                )
                return {
                    "delivered": True,
                    "status": "DELIVERED",
                    "message": f"SMS dispatched to {target_num} via Twilio (SID: {msg.sid[:8]}...)"
                }
            except Exception as e:
                return {"delivered": False, "status": "TWILIO_ERROR", "message": f"Twilio SMS error: {str(e)}"}

        return {
            "delivered": False,
            "status": "SMS_GATEWAY_NOT_CONFIGURED",
            "message": f"SMS prepared for {phone_number}. Provide Fast2SMS API key or Twilio credentials to dispatch real telecom SMS."
        }

    def _dispatch_phone_push(self, title: str, body: str) -> Dict[str, Any]:
        """
        Sends an instant real push notification to the user's phone via ntfy.sh.
        Zero credentials required - works on any phone with the free ntfy app.
        """
        if not requests or not self.ntfy_topic:
            return {"delivered": False, "status": "SKIPPED", "message": "Phone push skipped."}

        try:
            url = f"https://ntfy.sh/{self.ntfy_topic}"
            clean_title = title.encode("ascii", "ignore").decode("ascii").strip() or "RoadSentry AI Alert"
            headers = {
                "Title": clean_title[:200],
                "Priority": "urgent",
                "Tags": "rotating_light,ambulance,warning",
                "Click": "https://roadsentry.ai"
            }
            resp = requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=5)
            if resp.status_code == 200:
                return {
                    "delivered": True,
                    "status": "DELIVERED",
                    "topic": self.ntfy_topic,
                    "message": f"Mobile push alert sent to phone topic: {self.ntfy_topic}"
                }
            return {
                "delivered": False,
                "status": "FAILED",
                "message": f"Push server returned code {resp.status_code}"
            }
        except Exception as e:
            return {"delivered": False, "status": "ERROR", "message": str(e)}

    def _dispatch_webhook(
        self,
        incident: Dict[str, Any],
        location_str: str,
        subject: str,
        body: str
    ) -> Dict[str, Any]:
        """Dispatches an alert to Discord, Telegram, or any custom webhook URL if configured."""
        if not requests or not self.webhook_url:
            return {"delivered": False, "status": "SKIPPED", "message": "No webhook configured"}

        try:
            # Check if Discord webhook
            if "discord.com/api/webhooks" in self.webhook_url:
                payload = {
                    "content": f"**{subject}**\n```{body[:1800]}```"
                }
            else:
                payload = {
                    "event": "ROADSENTRY_ACCIDENT_SOS",
                    "incident": incident,
                    "location": location_str,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            resp = requests.post(self.webhook_url, json=payload, timeout=6)
            if resp.status_code in (200, 204):
                return {"delivered": True, "status": "DELIVERED", "message": "Webhook alert delivered successfully"}
            return {"delivered": False, "status": "FAILED", "message": f"Webhook returned {resp.status_code}"}
        except Exception as e:
            return {"delivered": False, "status": "ERROR", "message": str(e)}

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
Target Mobile: {self.recipient_phone}
"""
        sms_res = self._dispatch_sms(
            phone_number=self.recipient_phone,
            message=f"RoadSentry AI Test SOS\nStatus: Active\nTarget Mobile: {self.recipient_phone}\nTime: {timestamp}"
        )
        phone_res = self._dispatch_phone_push(
            title="RoadSentry AI TEST Alert",
            body=f"Test alert successful!\nTarget Mobile: {self.recipient_phone}\nTime: {timestamp}"
        )
        res = self._dispatch_email(subject, body, target)
        webhook_res = self._dispatch_webhook(
            incident={"incident_id": f"TEST-{int(time.time())}", "severity": "TEST"},
            location_str="Test Node",
            subject=subject,
            body=body
        )

        is_delivered = sms_res.get("delivered", False) or phone_res["delivered"] or res["delivered"] or webhook_res.get("delivered", False)
        messages = []
        if sms_res.get("delivered"):
            messages.append(f"📱 {sms_res['message']}")
        elif sms_res.get("status") not in ("SKIPPED", "SMS_GATEWAY_NOT_CONFIGURED"):
            messages.append(f"📱 SMS: {sms_res.get('message')}")

        if phone_res["delivered"]:
            messages.append(f"📲 {phone_res['message']}")
        if res["delivered"]:
            messages.append(res["message"])
        if webhook_res.get("delivered"):
            messages.append(f"🌐 {webhook_res['message']}")

        if not messages:
            messages.append(sms_res.get("message") or res["message"])

        final_status = "DELIVERED" if is_delivered else (sms_res.get("status") or res["status"])

        record = {
            "incident_id": f"TEST-{int(time.time())}",
            "timestamp": timestamp,
            "severity": "TEST",
            "location": "Local Node Test",
            "recipient_email": target,
            "recipient_phone": self.recipient_phone,
            "status": final_status,
            "delivered": is_delivered,
            "message": " | ".join(messages),
            "sms": sms_res,
            "phone_push": phone_res,
            "webhook": webhook_res
        }
        self.last_status = record
        self.dispatch_history.insert(0, record)
        return record

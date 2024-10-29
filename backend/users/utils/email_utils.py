# myapp/utils/email_utils.py
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.conf import settings
from django.core.mail.backends.smtp import EmailBackend  # Added this import
import socket
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def generate_verification_token(user):
    from django.contrib.auth.tokens import default_token_generator
    return default_token_generator.make_token(user)

def get_verification_link(user, domain):
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = generate_verification_token(user)
    return f"http://{domain}/api/verify-email/{uid}/{token}/"

def test_smtp_connection() -> Tuple[bool, str]:
    """Test SMTP connection and return status."""
    try:
        backend = EmailBackend(
            host=settings.EMAIL_HOST,
            port=settings.EMAIL_PORT,
            username=settings.EMAIL_HOST_USER,
            password=settings.EMAIL_HOST_PASSWORD,
            use_tls=settings.EMAIL_USE_TLS,
            timeout=5  # 5 seconds timeout
        )
        backend.open()
        backend.close()
        return True, "SMTP connection successful"
    except ImportError as e:
        return False, f"Email backend configuration error: {str(e)}"
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {str(e)}"
    except socket.timeout:
        return False, "SMTP connection timed out"
    except Exception as e:
        return False, f"SMTP connection failed: {str(e)}"

def send_verification_email(user, request) -> Tuple[bool, Optional[str]]:
    """
    Send verification email with comprehensive error handling.
    
    Returns:
        Tuple[bool, Optional[str]]: (success status, error message if any)
    """
    try:
        # Test SMTP connection first
        connection_success, connection_message = test_smtp_connection()
        if not connection_success:
            logger.error(f"SMTP Connection Test Failed: {connection_message}")
            return False, connection_message

        # Prepare email content
        domain = get_current_site(request).domain
        verification_link = get_verification_link(user, domain)
        subject = 'Verify your email'
        message = f'Hello {user.username},\n\nPlease click the following link to verify your email address:\n{verification_link}\n\nThank you!'
        from_email = settings.DEFAULT_FROM_EMAIL
        recipient_list = [user.email]

        # Log email attempt
        logger.info(f"Attempting to send verification email to {user.email}")

        # Send email with increased timeout
        send_mail(
            subject=subject,
            message=message,
            from_email=from_email,
            recipient_list=recipient_list,
            fail_silently=False,
        )
        
        logger.info(f"Successfully sent verification email to {user.email}")
        return True, None

    except socket.gaierror as e:
        error_msg = f"DNS resolution failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
        
    except ConnectionRefusedError as e:
        error_msg = f"Connection refused: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Failed to send verification email: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    
def handle_registration(user, request):
    email_sent, error_message = send_verification_email(user, request)
    if not email_sent:
        logger.warning(f"Registration completed but email failed: {error_message}")
        return {
            'message': 'Registration successful but verification email could not be sent.',
            'error_detail': error_message
        }
    return {'message': 'Registration successful. Please check your email for verification.'}   
# def send_verification_email(user, request):
#     try:
#         domain = get_current_site(request).domain
#         verification_link = get_verification_link(user, domain)
#         subject = 'Email Verification'
#         message = message = f'Hello {user.username},\n\nPlease click the following link to verify your email address:\n{verification_link}\n\nThank you!'
#         # send_mail(subject, message, 'no-reply@example.com', [user.email],fail_silently=False,)
#         from_email = settings.DEFAULT_FROM_EMAIL
#         recipient_list = [user.email]
        
#         send_mail(
#             subject=subject,
#             message=message,
#             from_email=from_email,
#             recipient_list=recipient_list,
#             fail_silently=False,
#         )
    
#         return True
#     except Exception as e:
#         logger.error(f"Failed to send verification email: {str(e)}")
#         return False


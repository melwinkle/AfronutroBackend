from django.db import IntegrityError, transaction
from rest_framework import generics, permissions, status, viewsets
from rest_framework.authtoken.views import ObtainAuthToken,APIView
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from .serializers import DietaryAssessmentSerializer, UserSerializer, UserProfileSerializer,PasswordResetRequestSerializer, PasswordResetConfirmSerializer,EducationalContentSerializer
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.utils import timezone
from datetime import timedelta
from .models import ActivityLevel, DietaryAssessment, DietaryPreference, HealthGoal, PasswordHistory,EducationalContent
from recipes.models import Ingredient
from recipes.serializers import IngredientSerializer
from django.contrib.auth.hashers import check_password
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.core.mail import send_mail
from .utils.email_utils import handle_registration,send_verification_email
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.cache import cache
import json
import logging

logger = logging.getLogger(__name__)

User = get_user_model()

class RegisterView(generics.CreateAPIView):
    """
    API view to register users with email verification.
    """
    serializer_class = UserSerializer
    permission_classes = (permissions.AllowAny,)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            with transaction.atomic():
                # Create user
                user = serializer.save()
                user.save()
                
                # Create token
                token = Token.objects.create(user=user)
                
                # Handle email verification
                email_result = handle_registration(user, request)
                
                # Prepare response
                response_data = {
                    'message': email_result.get('message'),
                    'user': UserSerializer(user, context=self.get_serializer_context()).data,
                    'token': token.key
                }
                
                # Add error detail if email sending failed
                if 'error_detail' in email_result:
                    response_data['email_error'] = email_result['error_detail']
                
                return Response(response_data, status=status.HTTP_201_CREATED)
                
        except IntegrityError as e:
            logger.warning(f"Registration failed: Email already exists - {request.data.get('email')}")
            return Response(
                {'error': 'A user with this email already exists.'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        except Exception as e:
            logger.error(f"Registration failed: {str(e)}")
            return Response(
                {'error': 'Registration failed. Please try again later.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ResendVerificationEmailAPIView(APIView):
    """
    API view to resend the email verification link.
    """
    permission_classes = (permissions.AllowAny,)

    def post(self, request):
        email = request.data.get('email')

        if not email:
            return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)

            if user.is_active:
                return Response({'error': 'User is already active'}, status=status.HTTP_400_BAD_REQUEST)

            # Send the verification email
            email_sent,error_message=send_verification_email(user, request)
            if email_sent:
                return Response({'message': 'Verification email sent successfully'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Failed to send verification email'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            

        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist'}, status=status.HTTP_400_BAD_REQUEST)

class VerifyEmailAPIView(APIView):
    """
    API view to verify the user's email address.
    """
    def get(self, request, uidb64, token):
        try:
            uid = force_bytes(urlsafe_base64_decode(uidb64))
            user = get_object_or_404(User, pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            user.is_active = True
            user.save()
            return Response({'message': 'Email verified successfully. You can now log in.'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Verification link is invalid or has expired.'}, status=status.HTTP_400_BAD_REQUEST)
            
class LoginView(ObtainAuthToken):
    """
    API view to login.
    """
    def post(self, request, *args, **kwargs):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(request, email=email, password=password)
        if user:
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user_id': user.pk,
                'email': user.email
            })
        return Response({'error': 'Invalid Credentials'}, status=status.HTTP_400_BAD_REQUEST)

class ProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = (permissions.IsAuthenticated,)

    def get_object(self):
        return self.request.user
    

class LogoutView(APIView):
    """
    API view to log out.
    """
    permission_classes = (permissions.IsAuthenticated,)

    def post(self, request):
        logging.debug(f"User: {request.user}, Auth: {request.auth}")
        if request.user.is_authenticated:
            request.user.auth_token.delete()
            return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "User not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)

class ChangePasswordView(APIView):
    """
    API view to change a password after every 30 days.
    """
    permission_classes = [permissions.IsAuthenticated,]

    def post(self, request):
        user = request.user
        current_password = request.data.get('current_password')
        new_password = request.data.get('new_password')

        if not user.check_password(current_password):
            return Response({'error': 'Current password is incorrect.'}, status=status.HTTP_400_BAD_REQUEST)

        if current_password == new_password:
            return Response({'error': 'New password must be different from the current password.'}, status=status.HTTP_400_BAD_REQUEST)

        # Check password history
        password_history = PasswordHistory.objects.filter(user=user).order_by('-created_at')[:5]
        for history in password_history:
            if check_password(new_password, history.password):
                return Response({'error': 'Password has been used recently.'}, status=status.HTTP_400_BAD_REQUEST)

        # Check if password was changed within the last 30 days
        if user.last_password_change > timezone.now() - timedelta(days=30):
            return Response({'error': 'Password can only be changed once every 30 days.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            validate_password(new_password, user)
        except ValidationError as e:
            return Response({'error': list(e.messages)}, status=status.HTTP_400_BAD_REQUEST)

        # Save old password to history
        PasswordHistory.objects.create(user=user, password=user.password)

        # Set new password
        user.set_password(new_password)
        user.save()

        return Response({'message': 'Password successfully changed.'}, status=status.HTTP_200_OK)


class PasswordResetRequestAPIView(APIView):
    """
    API view to request a password reset email.
    """
    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            try:
                user = User.objects.get(email=email)
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                domain = get_current_site(request).domain
                reset_link = f"http://{domain}/forgot-password/{uid}/{token}/"
                send_mail(
                    'Password Reset Request',
                    f'Click the link to reset your password: {reset_link}',
                    'no-reply@example.com',
                    [email],
                    fail_silently=False,
                )
                return Response({'message': 'Password reset link sent'}, status=status.HTTP_200_OK)
            except User.DoesNotExist:
                return Response({'error': 'User with this email does not exist'}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PasswordResetConfirmAPIView(APIView):
    """
    API view to confirm the password reset and set a new password.
    """
    def get(self, request, uidb64, token):
        try:
            uid = force_bytes(urlsafe_base64_decode(uidb64))
            user = get_object_or_404(User, pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            # This is where you can render a form or provide instructions
            return Response({'message': 'Reset your password using the POST request.'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'The link is invalid or has expired.'}, status=status.HTTP_400_BAD_REQUEST)
        
    def post(self, request, uidb64, token):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        if serializer.is_valid():
            new_password = serializer.validated_data['new_password']
            try:
                uid = force_str(urlsafe_base64_decode(uidb64))
                user = User.objects.get(pk=uid)
                if default_token_generator.check_token(user, token):
                    user.set_password(new_password)
                    user.save()
                    return Response({'message': 'Password has been reset'}, status=status.HTTP_200_OK)
                return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)
            except User.DoesNotExist:
                return Response({'error': 'User does not exist'}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

  


class EducationalContentListCreateView(APIView):
    """
    API view to list all educational content or create a new one.
    """
    permission_classes = [permissions.AllowAny]
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        content = EducationalContent.objects.all()
        serializer = EducationalContentSerializer(content, many=True, context={'request': request})
        return Response(serializer.data)

    def post(self, request,*args, **kwargs):
        serializer = EducationalContentSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response({
                'message': 'Content created successfully',
                'content': serializer.data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class EducationalContentDetailView(APIView):
    """
    API view to retrieve, update or delete an educational content.
    """
    permission_classes = [permissions.AllowAny]
    parser_classes = (MultiPartParser, FormParser)
    

    def get_object(self, content_id):
        try:
            return EducationalContent.objects.get(content_id=content_id)
        except EducationalContent.DoesNotExist:
            return None

    def get(self, request, content_id):
        content = self.get_object(content_id)
        if content is not None:
            serializer = EducationalContentSerializer(content, context={'request': request})
            return Response(serializer.data)
        return Response({'error': 'Content not found'}, status=status.HTTP_404_NOT_FOUND)

    def put(self, request, content_id):
        content = self.get_object(content_id)
        if content is not None:
            serializer = EducationalContentSerializer(content, data=request.data, partial=True, context={'request': request})
            if serializer.is_valid():
                serializer.save()
                return Response({
                    'message': 'Content updated successfully',
                    'content': serializer.data
                })
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response({'error': 'Content not found'}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, content_id):
        content = self.get_object(content_id)
        if content is not None:
            content.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response({'error': 'Content not found'}, status=status.HTTP_404_NOT_FOUND)

class EducationalContentByTypeView(APIView):
    """
    API view to list educational content by type.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        contenttype = request.query_params.get('type')
        print(f"Requested content type: {contenttype}")
        if contenttype:
            content = EducationalContent.objects.filter(content_type=contenttype)
            print(f"Number of items found: {content.count()}")
            serializer = EducationalContentSerializer(content, many=True, context={'request': request})
            return Response(serializer.data)
        return Response({'error': 'Content type parameter is required'}, status=status.HTTP_400_BAD_REQUEST)



# def get(self, request, content_id=None):
#         if content_id:
#             try:
#                 content = EducationalContent.objects.get(content_id=content_id)
#                 serializer = EducationalContentSerializer(content)
#                 return Response(serializer.data)
#             except EducationalContent.DoesNotExist:
#                 return Response(status=status.HTTP_404_NOT_FOUND)
#         else:
#             contents = EducationalContent.objects.all()
#             serializer = EducationalContentSerializer(contents, many=True)
#             return Response(serializer.data)












class AssessmentCalculationMixin:
    def calculate_tdee(self, weight, height, activity_levels, age, gender):
        # Calculate BMR using the Mifflin-St Jeor Equation
        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # Get the highest activity level factor
        activity_factors = {
            'SED': 1.2,
            'LIG': 1.375,
            'MOD': 1.55,
            'VER': 1.725,
            'EXT': 1.9
        }
        max_activity_factor = max(activity_factors[level] for level in activity_levels)

        # Calculate TDEE
        return bmr * max_activity_factor

    def calculate_bmi(self, weight, height):
        return weight / ((height / 100) ** 2)
    
    def generate_assessment_summary(self, data, tdee, bmi):
        summary = f"Based on your information, your Total Daily Energy Expenditure (TDEE) should be {tdee:.2f} calories. "
        summary += f"Your Body Mass Index (BMI) is {bmi:.2f}, which indicates that you "

        if bmi < 18.5:
            summary += "are underweight. "
        elif 18.5 <= bmi < 25:
            summary += "have a normal weight. "
        elif 25 <= bmi < 30:
            summary += "are overweight. "
        else:
            summary += "are obese. "

        # Health Goals Recommendations
        goals = data['health_goals']
        goal_recommendations = []

        if 'LOS' in goals and 'MUS' in goals:
            goal_recommendations.append(f"For weight loss and muscle gain, consider a daily calorie intake of {tdee:.0f} calories. Focus on strength training and include a mix of cardio.")
        elif 'LOS' in goals and 'GAI' in goals:
            goal_recommendations.append(f"For weight loss and muscle gain, aim for a calorie deficit while including high-protein foods.")
        elif 'LOS' in goals and 'MAI' in goals:
            goal_recommendations.append(f"For weight loss while maintaining your current weight, aim for a moderate calorie intake of {tdee:.0f} calories.")
        elif 'LOS' in goals and 'FIT' in goals:
            goal_recommendations.append(f"For weight loss and improved fitness, incorporate both cardio and strength training.Consider a daily calorie intake of {tdee:.0f} calories")
       

        elif 'GAI' in goals and 'MUS' in goals:
            goal_recommendations.append(f"For weight gain and muscle increase, aim for a daily calorie intake of {tdee:.0f} calories and focus on strength training.")
        elif 'GAI' in goals and 'MAI' in goals:
            goal_recommendations.append(f"For weight gain while maintaining your current weight, consume nutrient-dense foods.")
        elif 'GAI' in goals and 'FIT' in goals:
            goal_recommendations.append(f"For weight gain and improved fitness, focus on strength training and healthy fats.")
        
        
        elif 'MAI' in goals and 'FIT' in goals:
            goal_recommendations.append(f"To maintain your current weight while improving fitness, continue your routine and focus on a balanced diet.")
    

        elif 'FIT' in goals and 'MUS' in goals:
            goal_recommendations.append(f"To improve fitness while increasing muscle, ensure you include strength training and adequate protein intake.")
        elif 'FIT' in goals and 'LOS' in goals:
            goal_recommendations.append(f"To improve fitness while losing weight, combine cardiovascular exercise with strength training and maintain a caloric deficit.")

        elif 'LOS' in goals:

            goal_recommendations.append(f"For weight loss, consider a daily calorie intake of {tdee:.0f} calories. Incorporate cardio and strength training.")
        elif 'GAI' in goals:

            goal_recommendations.append(f"For weight gain, consider a daily calorie intake of {tdee:.0f} calories. Focus on strength training.")
        elif 'MAI' in goals:
            goal_recommendations.append(f"To maintain your current weight, aim for a daily calorie intake of approximately {tdee:.0f} calories.")
        elif 'FIT' in goals:
            goal_recommendations.append(f"To improve your fitness, engage in at least 150 minutes of moderate aerobic exercise each week.")
        elif 'MUS' in goals:

            goal_recommendations.append(f"For muscle increase, consider a daily calorie intake of {tdee:.0f} calories and prioritize strength training.")

        # Combine goal recommendations into the summary
        if goal_recommendations:
            summary += "\n- " + " ".join(goal_recommendations)

        # Dietary Preferences Recommendations
        dietary_preferences = data['dietary_preferences']
        summary += "Your dietary preferences include: " + ", ".join(dietary_preferences) + ". "

        if 'GLU' in dietary_preferences:
            summary += "\n- Consider gluten-free grains like quinoa or brown rice."
        if 'LAC' in dietary_preferences:
            summary += "\n- Opt for lactose-free dairy alternatives."
        if 'NUT' in dietary_preferences:
            summary += "\n- Avoid nuts; consider seeds for healthy fats."
        if 'SHE' in dietary_preferences:
            summary += "\n- Avoid shellfish and consider fish alternatives."
        if 'EGG' in dietary_preferences:
            summary += "\n- Choose egg substitutes for recipes."
        if 'SOY' in dietary_preferences:
            summary += "\n- Avoid soy products; consider alternatives like coconut yogurt."
        if 'PEA' in dietary_preferences:
            summary += "\n- Avoid peanuts; try sunflower seeds instead."
        if 'KOS' in dietary_preferences:
            summary += "\n- Ensure your food is certified kosher."
        if 'HAL' in dietary_preferences:
            summary += "\n- Choose halal-certified food options."
        if 'VEG' in dietary_preferences:
            summary += "\n- Include plant-based proteins like lentils and chickpeas."
        if 'VGT' in dietary_preferences:
            summary += "\n- Focus on plant-based foods, including dairy and eggs."
        if 'LSU' in dietary_preferences:
            summary += "\n- Limit sugar intake; choose naturally sweet foods."
        if 'DIA' in dietary_preferences:
            summary += "\n- Follow diabetic guidelines for carbohydrate intake."
        if 'SPI' in dietary_preferences:
            summary += "\n- Include moderate spice levels to your meals."
        if 'SWE' in dietary_preferences:
            summary += "\n- Limit sugary snacks; choose fruits for sweetness."
        if 'SAV' in dietary_preferences:
            summary += "\n- Focus on savory dishes with herbs and spices."
        if 'ORG' in dietary_preferences:
            summary += "\n- Choose organic produce and grains where possible."
        if 'HPR' in dietary_preferences:
            summary += "\n- Increase protein intake with lean meats and legumes."
        if 'LCA' in dietary_preferences:
            summary += "\n- Focus on low-carb options like vegetables and lean meats."
        if 'HFI' in dietary_preferences:
            summary += "\n- Include high-fiber foods such as whole grains and fruits."
        if 'KET' in dietary_preferences:
            summary += "\n- Consider a ketogenic approach with high fats and low carbs."
        if 'PAL' in dietary_preferences:
            summary += "\n- Focus on unprocessed foods and lean proteins."
        if 'DAI' in dietary_preferences:
            summary += "\n- Choose dairy alternatives like almond or oat milk."

        # Activity Level Recommendations
        activity_levels = data['activity_levels']
        summary += "Your activity levels include: " + ", ".join(activity_levels) + ". "

        if 'SED' in activity_levels:
            summary += "\n- Consider incorporating light exercises to break long periods of inactivity."
        if 'LIG' in activity_levels:
            summary += "\n- Engage in light activities such as walking or yoga."
        if 'MOD' in activity_levels:
            summary += "\n- Maintain a balanced routine of moderate exercise, including strength training."
        if 'VER' in activity_levels:
            summary += "\n- Incorporate more vigorous activities like running or high-intensity workouts."
        if 'EXT' in activity_levels:
            summary += "\n- Engage in intense training sessions and consider recovery strategies."
            
        # cuisins
        # Activity Level Recommendations
        cuisines = data['cuisine_preference']
        summary += "Your cuisine preferences include: " + ", ".join(cuisines) + ". "

        summary += "\n- We'll create meal plans that incorporate your liked ingredients and avoid your disliked ingredients where possible."

        return summary

class DietaryAssessmentView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    # change this to permission any 

    def get(self, request):
        """Get questionnaire and ingredient list"""
        questionnaire = {
            'dietary_preferences': DietaryPreference.choices,
            'activity_levels': ActivityLevel.choices,
            'health_goals': HealthGoal.choices,
        }
        ingredients = IngredientSerializer(Ingredient.objects.all(), many=True).data
        return Response({'questionnaire': questionnaire, 'ingredients': ingredients})

    def post(self, request):
        """Submit data from questionnaire"""
        self.permission_classes = [permissions.AllowAny]
        # Check permissions for this specific method
        self.check_permissions(request)
        data = request.data
        user = request.user

     

        # Calculate TDEE and BMI
        height = user.height  # Assuming you have a profile model with height
        weight = user.weight  # Assuming you have a profile model with weight
        tdee = self.calculate_tdee(weight, height, data['activity_levels'], user.age, user.gender)
        bmi = self.calculate_bmi(weight, height)
        
        if 'LOS' in data['health_goals']:
            tdee -= 500  # Adjust TDEE for weight loss
        elif 'GAI' or 'MUS' in data['health_goals']:
            tdee += 250  # Adjust TDEE for weight gain

        # Generate assessment summary
        assessment = self.generate_assessment_summary(data, tdee, bmi)

        # Prepare data for serializer
        assessment_data = {
            'user': user.id,
            'dietary_preferences': data['dietary_preferences'],
            'activity_levels': data['activity_levels'],
            'health_goals': data['health_goals'],
            'liked_ingredients': data['liked_ingredients'],
            'disliked_ingredients': data['disliked_ingredients'],
            'cuisine_preference':data['cuisine_preference'],
            'tdee': tdee,
            'bmi': bmi,
            'assessment': assessment,
        }

        serializer = DietaryAssessmentSerializer(data=assessment_data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response({
                'message': 'Dietary assessment created successfully.',
                'tdee': tdee,
                'bmi': bmi,
                'assessment': assessment,
                'data': serializer.data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request):
        """Update preferences"""
        try:
            assessment = DietaryAssessment.objects.get(user=request.user)
        except DietaryAssessment.DoesNotExist:
            return Response({'error': 'DietaryAssessment not found'}, status=status.HTTP_404_NOT_FOUND)

        # Prepare the data for the serializer, excluding many-to-many fields
        assessment_data = request.data.copy()  # Create a mutable copy of the request data

        # Use a serializer for the non-many-to-many fields
        serializer = DietaryAssessmentSerializer(assessment, data=assessment_data, partial=True, context={'request': request})
        
        if serializer.is_valid():
            # Save non-many-to-many fields first
            serializer.save()

            # Handle many-to-many fields
            if 'disliked_ingredients' in assessment_data:
                # Get actual Ingredient objects based on names
                disliked_ingredients = assessment_data['disliked_ingredients']
                ingredient_objects = Ingredient.objects.filter(name__in=disliked_ingredients)

                # Check if any ingredients are missing
                if len(ingredient_objects) != len(disliked_ingredients):
                    return Response({"error": "One or more disliked ingredients do not exist."}, status=status.HTTP_400_BAD_REQUEST)

                # Set the many-to-many relationship
                assessment.disliked_ingredients.set(ingredient_objects)

            if 'liked_ingredients' in assessment_data:
                # Get actual Ingredient objects based on names
                liked_ingredients = assessment_data['liked_ingredients']
                ingredient_objects = Ingredient.objects.filter(name__in=liked_ingredients)

                # Check if any ingredients are missing
                if len(ingredient_objects) != len(liked_ingredients):
                    return Response({"error": "One or more liked ingredients do not exist."}, status=status.HTTP_400_BAD_REQUEST)

                # Set the many-to-many relationship
                assessment.liked_ingredients.set(ingredient_objects)

            return Response(serializer.data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    
    def delete(self, request):
        """Delete dietary assessment"""
        try:
            assessment = DietaryAssessment.objects.get(user=request.user)
            assessment.delete()
            return Response({'message': 'Dietary assessment deleted successfully.'}, status=status.HTTP_204_NO_CONTENT)
        except DietaryAssessment.DoesNotExist:
            return Response({'error': 'DietaryAssessment not found'}, status=status.HTTP_404_NOT_FOUND)

    def calculate_tdee(self, weight, height, activity_levels, age, gender):
        # Calculate BMR using the Mifflin-St Jeor Equation
        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # Get the highest activity level factor
        activity_factors = {
            'SED': 1.2,
            'LIG': 1.375,
            'MOD': 1.55,
            'VER': 1.725,
            'EXT': 1.9
        }
        max_activity_factor = max(activity_factors[level] for level in activity_levels)

        # Calculate TDEE
        tdee = bmr * max_activity_factor

        return tdee

    def calculate_bmi(self, weight, height):
        return weight / ((height / 100) ** 2)

    def generate_assessment_summary(self, data, tdee, bmi):
        summary = f"Based on your information, your Total Daily Energy Expenditure (TDEE) should be {tdee:.2f} calories. "
        summary += f"Your Body Mass Index (BMI) is {bmi:.2f}, which indicates that you "

        if bmi < 18.5:
            summary += "are underweight. "
        elif 18.5 <= bmi < 25:
            summary += "have a normal weight. "
        elif 25 <= bmi < 30:
            summary += "are overweight. "
        else:
            summary += "are obese. "

        # Health Goals Recommendations
        goals = data['health_goals']
        goal_recommendations = []

        if 'LOS' in goals and 'MUS' in goals:
            goal_recommendations.append(f"For weight loss and muscle gain, consider a daily calorie intake of {tdee:.0f} calories. Focus on strength training and include a mix of cardio.")
        elif 'LOS' in goals and 'GAI' in goals:
            goal_recommendations.append(f"For weight loss and muscle gain, aim for a calorie deficit while including high-protein foods.")
        elif 'LOS' in goals and 'MAI' in goals:
            goal_recommendations.append(f"For weight loss while maintaining your current weight, aim for a moderate calorie intake of {tdee:.0f} calories.")
        elif 'LOS' in goals and 'FIT' in goals:
            goal_recommendations.append(f"For weight loss and improved fitness, incorporate both cardio and strength training.Consider a daily calorie intake of {tdee:.0f} calories")
       

        elif 'GAI' in goals and 'MUS' in goals:
            goal_recommendations.append(f"For weight gain and muscle increase, aim for a daily calorie intake of {tdee:.0f} calories and focus on strength training.")
        elif 'GAI' in goals and 'MAI' in goals:
            goal_recommendations.append(f"For weight gain while maintaining your current weight, consume nutrient-dense foods.")
        elif 'GAI' in goals and 'FIT' in goals:
            goal_recommendations.append(f"For weight gain and improved fitness, focus on strength training and healthy fats.")
        
        
        elif 'MAI' in goals and 'FIT' in goals:
            goal_recommendations.append(f"To maintain your current weight while improving fitness, continue your routine and focus on a balanced diet.")
    

        elif 'FIT' in goals and 'MUS' in goals:
            goal_recommendations.append(f"To improve fitness while increasing muscle, ensure you include strength training and adequate protein intake.")
        elif 'FIT' in goals and 'LOS' in goals:
            goal_recommendations.append(f"To improve fitness while losing weight, combine cardiovascular exercise with strength training and maintain a caloric deficit.")

        elif 'LOS' in goals:

            goal_recommendations.append(f"For weight loss, consider a daily calorie intake of {tdee:.0f} calories. Incorporate cardio and strength training.")
        elif 'GAI' in goals:

            goal_recommendations.append(f"For weight gain, consider a daily calorie intake of {tdee:.0f} calories. Focus on strength training.")
        elif 'MAI' in goals:
            goal_recommendations.append(f"To maintain your current weight, aim for a daily calorie intake of approximately {tdee:.0f} calories.")
        elif 'FIT' in goals:
            goal_recommendations.append(f"To improve your fitness, engage in at least 150 minutes of moderate aerobic exercise each week.")
        elif 'MUS' in goals:

            goal_recommendations.append(f"For muscle increase, consider a daily calorie intake of {tdee:.0f} calories and prioritize strength training.")

        # Combine goal recommendations into the summary
        if goal_recommendations:
            summary += "\n- " + " ".join(goal_recommendations)

        # Dietary Preferences Recommendations
        dietary_preferences = data['dietary_preferences']
        summary += "Your dietary preferences include: " + ", ".join(dietary_preferences) + ". "

        if 'GLU' in dietary_preferences:
            summary += "\n- Consider gluten-free grains like quinoa or brown rice."
        if 'LAC' in dietary_preferences:
            summary += "\n- Opt for lactose-free dairy alternatives."
        if 'NUT' in dietary_preferences:
            summary += "\n- Avoid nuts; consider seeds for healthy fats."
        if 'SHE' in dietary_preferences:
            summary += "\n- Avoid shellfish and consider fish alternatives."
        if 'EGG' in dietary_preferences:
            summary += "\n- Choose egg substitutes for recipes."
        if 'SOY' in dietary_preferences:
            summary += "\n- Avoid soy products; consider alternatives like coconut yogurt."
        if 'PEA' in dietary_preferences:
            summary += "\n- Avoid peanuts; try sunflower seeds instead."
        if 'KOS' in dietary_preferences:
            summary += "\n- Ensure your food is certified kosher."
        if 'HAL' in dietary_preferences:
            summary += "\n- Choose halal-certified food options."
        if 'VEG' in dietary_preferences:
            summary += "\n- Include plant-based proteins like lentils and chickpeas."
        if 'VGT' in dietary_preferences:
            summary += "\n- Focus on plant-based foods, including dairy and eggs."
        if 'LSU' in dietary_preferences:
            summary += "\n- Limit sugar intake; choose naturally sweet foods."
        if 'DIA' in dietary_preferences:
            summary += "\n- Follow diabetic guidelines for carbohydrate intake."
        if 'SPI' in dietary_preferences:
            summary += "\n- Include moderate spice levels to your meals."
        if 'SWE' in dietary_preferences:
            summary += "\n- Limit sugary snacks; choose fruits for sweetness."
        if 'SAV' in dietary_preferences:
            summary += "\n- Focus on savory dishes with herbs and spices."
        if 'ORG' in dietary_preferences:
            summary += "\n- Choose organic produce and grains where possible."
        if 'HPR' in dietary_preferences:
            summary += "\n- Increase protein intake with lean meats and legumes."
        if 'LCA' in dietary_preferences:
            summary += "\n- Focus on low-carb options like vegetables and lean meats."
        if 'HFI' in dietary_preferences:
            summary += "\n- Include high-fiber foods such as whole grains and fruits."
        if 'KET' in dietary_preferences:
            summary += "\n- Consider a ketogenic approach with high fats and low carbs."
        if 'PAL' in dietary_preferences:
            summary += "\n- Focus on unprocessed foods and lean proteins."
        if 'DAI' in dietary_preferences:
            summary += "\n- Choose dairy alternatives like almond or oat milk."

        # Activity Level Recommendations
        activity_levels = data['activity_levels']
        summary += "Your activity levels include: " + ", ".join(activity_levels) + ". "

        if 'SED' in activity_levels:
            summary += "\n- Consider incorporating light exercises to break long periods of inactivity."
        if 'LIG' in activity_levels:
            summary += "\n- Engage in light activities such as walking or yoga."
        if 'MOD' in activity_levels:
            summary += "\n- Maintain a balanced routine of moderate exercise, including strength training."
        if 'VER' in activity_levels:
            summary += "\n- Incorporate more vigorous activities like running or high-intensity workouts."
        if 'EXT' in activity_levels:
            summary += "\n- Engage in intense training sessions and consider recovery strategies."
        
        cuisines = data['cuisine_preference']
        summary += "Your cuisine preferences include: " + ", ".join(cuisines) + ". "

        summary += "\n- We'll create meal plans that incorporate your liked ingredients and avoid your disliked ingredients where possible."

        return summary

class RecalculateAssessmentView(AssessmentCalculationMixin,APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            assessment = DietaryAssessment.objects.get(user=request.user)
            
            # Get current user stats
            height = request.user.height
            weight = request.user.weight
            
            # Recalculate TDEE and BMI
            tdee = self.calculate_tdee(
                weight, 
                height, 
                assessment.activity_levels,
                request.user.age,
                request.user.gender
            )
            bmi = self.calculate_bmi(weight, height)
            
            # Adjust TDEE based on existing health goals
            if 'LOS' in assessment.health_goals:
                tdee -= 500
            elif 'GAI' in assessment.health_goals or 'MUS' in assessment.health_goals:
                tdee += 250
                
            # Generate new assessment summary
            assessment_data = {
                'dietary_preferences': assessment.dietary_preferences,
                'activity_levels': assessment.activity_levels,
                'health_goals': assessment.health_goals,
                'liked_ingredients': assessment.liked_ingredients.all(),
                'disliked_ingredients': assessment.disliked_ingredients.all(),
                'cuisine_preference':assessment.cuisine_preference
            }
            
            new_summary = self.generate_assessment_summary(
                assessment_data, tdee, bmi
            )
            
            # Update assessment
            assessment.tdee = tdee
            assessment.bmi = bmi
            assessment.assessment = new_summary
            assessment.save()
            
            return Response({
                'message': 'Assessment recalculated successfully',
                'tdee': tdee,
                'bmi': bmi,
                'assessment': new_summary
            })
            
        except DietaryAssessment.DoesNotExist:
            return Response(
                {'error': 'No assessment found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
class DietaryAssessmentRetrieveView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            assessment = DietaryAssessment.objects.get(user=request.user)
        except DietaryAssessment.DoesNotExist:
            return Response(
                {"error": "DietaryAssessment not found or you don't have permission to view it."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = DietaryAssessmentSerializer(assessment)
        return Response(serializer.data)


# class GenerateMealPlanView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     @staticmethod
#     def safe_load(data):
#         if isinstance(data, str):
#             try:
#                 return json.loads(data)
#             except json.JSONDecodeError:
#                 return data
#         return data

#     def post(self, request):
#         try:
#             dietary_assessment = DietaryAssessment.objects.get(user=request.user)
#         except DietaryAssessment.DoesNotExist:
#             return Response({"error": "Dietary assessment not found."}, status=status.HTTP_404_NOT_FOUND)

#         # Retrieve the fitted recommender from cache
#         recommender = cache.get('hybrid_recommender')
        
#         if not recommender:
#             fit_recommender()
#             cache.set('recommender_fitted', True)
#             cache.set('hybrid_recommender',recommender)
#             # Check if the recommender has been fitted
               
#         if not recommender:
#             return Response({"error": "Recommender system not ready. Please try again later."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


#         # Prepare user profile data
#         user_profile = {
#             'user_id': request.user.id,
#             'age': request.user.age,
#             'gender': request.user.gender,
#             'weight': request.user.weight,
#             'height': request.user.height,
#             'activity_level': self.safe_load(dietary_assessment.activity_levels),
#             'tdee': dietary_assessment.tdee,
#             'bmi': dietary_assessment.bmi,
#             'liked_ingredients': self.safe_load(dietary_assessment.liked_ingredients),
#             'disliked_ingredients': self.safe_load(dietary_assessment.disliked_ingredients),
#             'dietary_preferences': self.safe_load(dietary_assessment.dietary_preferences),
#             'health_goal': self.safe_load(dietary_assessment.health_goals)
#         }

#         # Get recommendations using the comprehensive user profile
#         recommendations = recommender.get_recommendations(user_profile)

#         meal_plan = MealPlan(
#             user=request.user,
#             name="AI Generated Meal Plan",
#             description="Automatically generated based on your dietary assessment.",
#             status=MealPlan.DRAFT
#         )
#         meal_plan.save()  # Save to get an ID

#         for meal_type, recipes in recommendations.items():
#             for recipe_name in recipes:
#                 try:
#                     recipe = Recipe.objects.get(name=recipe_name)
#                     meal_plan.meals.add(recipe)
#                 except Recipe.DoesNotExist:
#                     # Log this error or handle it as appropriate for your application
#                     pass

#         # Store in cache for 24 hours
#         cache.set(f'meal_plan_draft_{meal_plan.meal_plan_id}', meal_plan, 60*60*24)

#         serializer = MealPlanSerializer(meal_plan)
#         return Response(serializer.data, status=status.HTTP_201_CREATED)
    
#     def get(self, request, meal_plan_id=None):
#         # If `meal_plan_id` is provided, retrieve specific meal plan by ID
#         if meal_plan_id:
#             try:
#                 meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
#                 serializer = MealPlanSerializer(meal_plan)
#                 return Response(serializer.data, status=status.HTTP_200_OK)
#             except MealPlan.DoesNotExist:
#                 return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)
        
#         # Otherwise, retrieve all meal plans for the authenticated user
#         meal_plans = MealPlan.objects.filter(user=request.user)
#         serializer = MealPlanSerializer(meal_plans, many=True)
#         return Response(serializer.data, status=status.HTTP_200_OK)
    
#     def delete(self, request, meal_plan_id):
#         try:
#             mealplan = MealPlan.objects.get(user=request.user, meal_plan_id=meal_plan_id)
#             mealplan.delete()
#             return Response(status=status.HTTP_204_NO_CONTENT)
#         except MealPlan.DoesNotExist:
#             return Response({'error': 'Meal plan not found'}, status=status.HTTP_404_NOT_FOUND)

# class SaveMealPlanView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, meal_plan_id):
#         meal_plan = cache.get(f'meal_plan_draft_{meal_plan_id}')
        
#         if not meal_plan:
#             return Response({"error": "Draft meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

#         meal_plan.status = MealPlan.SAVED
#         meal_plan.save()

#         # Remove from cache
#         cache.delete(f'meal_plan_draft_{meal_plan_id}')

#         serializer = MealPlanSerializer(meal_plan)
#         return Response(serializer.data, status=status.HTTP_200_OK)

# class CustomizeMealPlanView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     def put(self, request, meal_plan_id):
#         meal_plan = cache.get(f'meal_plan_draft_{meal_plan_id}')
        
#         if not meal_plan:
#             try:
#                 meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
#             except MealPlan.DoesNotExist:
#                 return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

#         serializer = MealPlanSerializer(meal_plan, data=request.data, partial=True)
#         if serializer.is_valid():
#             updated_meal_plan = serializer.save()
            
#             # If it's a draft, update the cache
#             if updated_meal_plan.status == MealPlan.DRAFT:
#                 cache.set(f'meal_plan_draft_{meal_plan_id}', updated_meal_plan, 60*60*24)
            
#             return Response(serializer.data)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
# class NutritionalSummaryView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     def get(self, request, meal_plan_id):
#         try:
#             meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
#         except MealPlan.DoesNotExist:
#             return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

#         try:
#             dietary_assessment = DietaryAssessment.objects.get(user=request.user)
#         except DietaryAssessment.DoesNotExist:
#             return Response({"error": "Dietary assessment not found."}, status=status.HTTP_404_NOT_FOUND)

#         nutritional_composition = meal_plan.calculate_nutritional_composition()
#         tdee = dietary_assessment.tdee

#         summary = {
#             "total_calories": nutritional_composition["calories"],
#             "total_protein": nutritional_composition["protein"],
#             "total_carbs": nutritional_composition["carbs"],
#             "total_fat": nutritional_composition["fat"],
#             "tdee": tdee,
#             "calorie_difference": nutritional_composition["calories"] - tdee
#         }

#         return Response(summary)

# class SetTagsView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, meal_plan_id):
#         try:
#             meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
#         except MealPlan.DoesNotExist:
#             return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

#         try:
#             dietary_assessment = DietaryAssessment.objects.get(user=request.user)
#         except DietaryAssessment.DoesNotExist:
#             return Response({"error": "Dietary assessment not found."}, status=status.HTTP_404_NOT_FOUND)

#         tags = {
#             "health_goals": dietary_assessment.health_goals,
#             "dietary_preferences": dietary_assessment.dietary_preferences
#         }

#         meal_plan.tags = tags
#         meal_plan.save()

#         serializer = MealPlanSerializer(meal_plan)
#         return Response(serializer.data)



class ActivityLevelView(APIView):
    def get(self, request):
        choices = ActivityLevel.choices
        serialized_choices = [
            {"value": level[0], "display_name": level[1]} for level in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)

class HealthGoalView(APIView):
    def get(self, request):
        choices = HealthGoal.choices
        serialized_choices = [
            {"value": goal[0], "display_name": goal[1]} for goal in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)

class DietaryPreferenceView(APIView):
    def get(self, request):
        choices = DietaryPreference.choices
        serialized_choices = [
            {"value": preference[0], "display_name": preference[1]} for preference in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)


# python manage.py update_recommender
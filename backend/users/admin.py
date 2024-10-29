from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _
from .models import User

class CustomUserAdmin(UserAdmin):
    model = User
    list_display = ('email', 'username','is_staff', 'is_active','is_verified','age','gender','height','weight','last_password_change',)
    list_filter = ('email', 'is_staff', 'is_active','age','gender',)
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Personal info'), {'fields': ('first_name', 'last_name', 'username','bio','age','gender','height','weight','last_password_change',)}),
        (_('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )
    search_fields = ('email','username',)
    ordering = ('email',)

admin.site.register(User, CustomUserAdmin)
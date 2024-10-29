from django.apps import AppConfig
from django.core.cache import cache

class MealPlannerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'meal_planner'
    
    def ready(self):
        if not cache.get('recommender_fitted'):
            from .tasks import fit_recommender_task
            fit_recommender_task.delay()
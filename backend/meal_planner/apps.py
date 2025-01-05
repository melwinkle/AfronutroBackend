from django.apps import AppConfig
from django.core.cache import cache

class MealPlannerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'meal_planner'
    
    def ready(self):
        # Force clear cache when application starts
        cache.delete('hybrid_recommender')
        cache.delete('recommender_fitted')
        cache.delete('evaluation_metrics')
        cache.delete('last_training_time')
        
        # Force retrain
        if True:  # Change this to True to force retraining
            from .tasks import fit_recommender_task
            fit_recommender_task.delay()
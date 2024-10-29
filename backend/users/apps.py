from django.apps import AppConfig
# from django.db.models.signals import post_migrate
# from django.core.cache import cache

class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    # def ready(self):
    #     from django.conf import settings
    #     from .ml_utils import fit_recommender

    #     # if not settings.TESTING:  # Don't run during tests
    #     #     post_migrate.connect(self.fit_recommender_once, sender=self)

    # def fit_recommender_once(sender, **kwargs):
    #     if not cache.get('recommender_fitted'):
    #         from .ml_utils import fit_recommender
    #         fit_recommender()
    #         cache.set('recommender_fitted', True)
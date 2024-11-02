import csv
from django.core.management.base import BaseCommand
from recipes.models import Recipe, Ingredient, NutritionalInformation
from django.db import transaction

class Command(BaseCommand):
    help = 'Load recipe and ingredient data from CSV files'

    def add_arguments(self, parser):
        parser.add_argument('recipes_file', type=str, help='meal_planner/training_data/recipes.csv')
        parser.add_argument('ingredients_file', type=str, help='meal_planner/training_data/ingredients.csv')

    def clean_array_columns(self, row, array_columns):
        """
        Clean specific columns in the row by removing surrounding characters and formatting as PostgreSQL arrays.
        """
        for col in array_columns:
            if row[col].startswith('[') and row[col].endswith(']'):
                # Remove surrounding square brackets and quotes, and convert to PostgreSQL array format
                row[col] = "{" + row[col][1:-1].replace("'", "").replace('"', '').strip() + "}"
        return row

    def handle(self, *args, **options):
        recipes_file = options['recipes_file']
        ingredients_file = options['ingredients_file']

        # Define which columns in recipes contain arrays
        recipe_array_columns = ['ingredients', 'tags','cuisine','meal_type','dish_type']
        ingredient_array_columns = ['minerals', 'vitamins', 'substitutes']

        self.stdout.write('Starting to load data...')

        try:
            with transaction.atomic():
                # First load recipes
                self.stdout.write('Loading recipes...')
                with open(recipes_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    recipes_created = 0
                    for row in reader:
                        # Clean array columns for recipes
                        row = self.clean_array_columns(row, recipe_array_columns)

                        # Create nutritional information
                        nutrition = NutritionalInformation.objects.create(
                            calories=float(row.get('calories', 0)),
                            protein=float(row.get('protein', 0)),
                            carbs=float(row.get('carbs', 0)),
                            fat=float(row.get('fat', 0)),
                            fiber=float(row.get('fiber', 0))
                        )

                        # Create recipe
                        recipe = Recipe.objects.create(
                            name=row['name'],
                            ingredients=row['ingredients'],  # Now in PostgreSQL array format
                            cuisine=row['cuisine'],
                            recipe_info=row['recipe_info'],
                            tags=row['tags'],  # Now in PostgreSQL array format
                            meal_type=row['meal_type'],
                            dish_type=row['dish_type'],
                            vegan=row.get('vegan', '').lower() == 'true',
                            vegetarian=row.get('vegetarian', '').lower() == 'true',
                            gluten_free=row.get('gluten_free', '').lower() == 'true',
                            pescatarian=row.get('pescatarian', '').lower() == 'true',
                            halal=row.get('halal', '').lower() == 'true',
                            nutrition=nutrition
                        )
                        recipes_created += 1

                        if recipes_created % 100 == 0:  # Progress update every 100 recipes
                            self.stdout.write(f'Loaded {recipes_created} recipes...')

                self.stdout.write(f'Successfully loaded {recipes_created} recipes')

                # Then load ingredients
                self.stdout.write('Loading ingredients...')
                with open(ingredients_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    ingredients_created = 0
                    for row in reader:
                        # Clean array columns for ingredients
                        row = self.clean_array_columns(row, ingredient_array_columns)

                        # Create ingredient
                        ingredient = Ingredient.objects.create(
                            name=row['name'],
                            calories=row.get('calories', 0),
                            protein=row.get('protein', 0),
                            carbs=row.get('carbs', 0),
                            fat=row.get('fat', 0),
                            minerals=row['minerals'],  # Now in PostgreSQL array format
                            vitamins=row['vitamins'],  # Now in PostgreSQL array format
                        )
                        # Handle the substitutes ManyToMany field
                        substitutes_list = row['substitutes'].strip("{}").split(",") if row['substitutes'] else []
                        substitutes_list = [substitute.strip() for substitute in substitutes_list]
                        
                        # Assuming 'name' is unique in the Ingredient model and you are linking by name
                        substitutes_qs = Ingredient.objects.filter(name__in=substitutes_list)
                        ingredient.substitutes.set(substitutes_qs)
                        ingredients_created += 1

                self.stdout.write(f'Successfully loaded {ingredients_created} ingredients')

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error loading data: {str(e)}'))
            raise e

        self.stdout.write(self.style.SUCCESS('Data loading completed successfully!'))

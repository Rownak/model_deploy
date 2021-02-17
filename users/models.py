from django.db import models

# Create your models here.
from django.contrib.auth.models import User


class Profile(models.Model):
	# if user get deleted also delet the profile
	user = models.OneToOneField(User, on_delete=models.CASCADE)

	# What to display if user is printed
	def __str__(self):
		return f'{self.user.username} Profile'

	#
	def save(self):
		#run save method of parent class 
		super().save()
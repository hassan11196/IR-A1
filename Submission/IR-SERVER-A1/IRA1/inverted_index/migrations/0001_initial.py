# Generated by Django 3.0.4 on 2020-03-07 16:49

from django.db import migrations, models
import picklefield.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='InvertedIndexModel',
            fields=[
                ('data', picklefield.fields.PickledObjectField(editable=False)),
                ('status', models.BooleanField(default=False)),
                ('id', models.DateTimeField(auto_now_add=True, primary_key=True, serialize=False)),
            ],
        ),
    ]

from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings
import os

admin.autodiscover()
MEDIA_ROOT = os.path.join(settings.BASE_DIR, 'demo/images_pool')
urlpatterns = patterns('',
    # Examples:
    url(r'^$', 'demo.views.index', name='home'),
    url(r'^index', 'demo.views.index'),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^demo/$', 'demo.views.retrieve'),
    url(r'^upload_img/$', 'demo.views.upload_img'),
    #url(r'^image/(?P<path>.)*$', 'demo.views.image')
    #url(r'^images_pool/(?P<path>.*)$', 'demo.views.image', {'document_root': WEB_ROOT})
) + static('/demo/images_pool/', document_root=MEDIA_ROOT)


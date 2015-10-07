from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings
import os

admin.autodiscover()
MEDIA_ROOT = os.path.join(settings.BASE_DIR, 'images_pool')
urlpatterns = patterns('',
    # Examples:
    url(r'^$', 'demo.views.index', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    #url(r'^demo/$', 'demo.views.demo'),
    #url(r'^image/(?P<path>.)*$', 'demo.views.image')
    #url(r'^images_pool/(?P<path>.*)$', 'demo.views.image', {'document_root': WEB_ROOT})
) + static('/images_pool/', document_root=MEDIA_ROOT)


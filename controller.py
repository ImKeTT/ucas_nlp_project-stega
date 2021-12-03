#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: controller.py
@author: ImKe at 2021/12/1
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import web
import web_utils
from main_func import main

render = web.template.render('templates')

urls = (
    '/', 'index'
)
class index:
    def GET(self):
        data = {
            'form': web_utils.FORM_INIT,
            'header': web_utils.HEADER,
            'landing': HC.get_landing_data(),
            'footer': web_utils.FOOTER,
        }
        return render.index(data=data)

# import web
#
# urls = (
#     '/(.*)', 'hello'
# )
#
#
#
# class hello:
#     def GET(self, name):
#         i = web.input(times=1)
#         if not name:
#             name = 'world'
#         for c in range(int(i.times)):
#             print('Hello,', name + '!')
#         return 'Hello, ' + name + '!'


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()


#!/usr/bin/python

import PyV8
ctxt = PyV8.JSContext()
ctxt.enter()
ctxt.eval(open("keyboard_input_manager.js").read())
ctxt.eval("var template = 'Javascript in Python is {{ opinion }}';")

import random
opinion = random.choice(["cool","great","nice","insane"])
rendered = ctxt.eval("Mustache.to_html(template, { opinion: '%s' })" % (opinion, ))
print rendered


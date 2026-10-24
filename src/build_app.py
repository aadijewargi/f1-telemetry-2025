import json, sys
data = open("analysis_out/app_data.json").read().replace("</", "<\\/")
tpl = open("app/template.html").read()
body = tpl.replace("__DATA__", data)
open("app/app_artifact.html", "w").write(body)
head = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>'
# in the standalone page the <meta>/<title>/<link>/<style> sit inside body, which browsers hoist correctly
open("app/index.html", "w").write(head + body + "</body></html>")
print(len(body) / 1e6, "MB")

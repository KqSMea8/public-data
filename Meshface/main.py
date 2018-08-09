import web

urls = (
    '/', 'index'
)

class index:
    def GET(self):
        return render.index()

render = web.template.render('templates/',base='layout') 
app = web.application(urls, globals())

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
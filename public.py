import nest_asyncio
from pyngrok import ngrok
import uvicorn
from main import app
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
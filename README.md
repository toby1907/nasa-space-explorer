# 🛰️ NASA Space Explorer Agent

A fully A2A protocol compliant AI agent that delivers daily Astronomy Pictures from NASA's APOD API directly to Telex.im chat platform.

## 🚀 Features

- **Daily Astronomy Pictures**: Get today's NASA Astronomy Picture of the Day
- **Random Space Images**: Explore random images from NASA's extensive archive
- **Space Facts**: Learn fascinating facts about our universe
- **Multiple Commands**: Support for various space-related queries
- **Telex Integration**: Seamless integration with Telex.im chat platform

## 📋 Available Commands

- `"today's image"` - Today's Astronomy Picture of the Day
- `"random image"` - Random space image from NASA's archive
- `"yesterday's image"` - Yesterday's astronomy picture
- `"space fact"` or `"random fact"` - Interesting space facts
- `"help"` - Show help message

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **Protocol**: A2A (Agent-to-Agent) compliant
- **API**: NASA APOD API
- **Deployment**: Heroku-ready
- **Format**: JSON-RPC 2.0

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- NASA API Key (optional - uses DEMO_KEY by default)

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd nasa-space-explorer

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NASA_API_KEY=your_nasa_api_key_here
export PORT=8000

# Run the application
python main.py
```

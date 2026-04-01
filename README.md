# prompt-validator
A FastAPI-based microservice that implements an prompt validation and improvements using language models and graph-based conversation flow.
## :star2: Features
- FastAPI-powered RESTful API
- Configurable logging system
- Chat controller for managing conversations
## :clipboard: Prerequisites
- Python 3.12+
## :rocket: Getting Started
### Local Development Setup
1. Create a virtual environment:
```bash
python -m venv ./venv
```
2. Activate the virtual environment:
```bash
# On macOS/Linux
source ./venv/bin/activate
# On Windows
.\venv\Scripts\activate
```
3. Install dependencies:
```bash
# For development
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
4. Set up environment variables:
```bash
# Copy the sample env file
cp .env-sample .env
# Edit .env with your configuration
```
### :whale: Docker Setup
1. Build the Docker image:
```bash
docker build -t x42-prompt-validator .
```
2. Run using docker-compose:
```bash
docker-compose up
```
## :wrench: Configuration
- Configuration files are located in the `config/` directory
- Logging configuration: `config/log-config.yaml`
- Application configuration can be modified through environment variables
## :rocket: Running the Application
### Local Development
Start the FastAPI application:
```bash
uvicorn src.chat.controller.chat_controller:app --host 0.0.0.0 --port 21120
```
The application will be available at:
- API: `http://127.0.0.1:21120`
- Swagger Documentation: `http://127.0.0.1:21120/docs`
- ReDoc Documentation: `http://127.0.0.1:21120/redoc`
## :file_folder: Project Structure
```
prompt-validator/
├── src/                    # Source code
│   ├── chat/              # Chat related modules
│   ├── common/            # Shared utilities and configurations
│   └── main.py           # Application entry point
├── config/                # Configuration files
└── requirements.txt  # Production dependencies
```
## :memo: API Documentation
Once the application is running, you can access:
- Swagger UI: `http://127.0.0.1:21120/docs`
- ReDoc: `http://127.0.0.1:21120/redoc`
## :handshake: Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
## :page_facing_up: License
This project is proprietary and confidential. All rights reserved.
## :sos: Support
For support and questions, please contact the development team.

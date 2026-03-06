"""
MammoScan AI - Breast Cancer Detection Web Application
Entry point for the Flask application.
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Use 'stat' reloader instead of 'watchdog' to prevent Flask from
    # watching TensorFlow's internal files and restarting in an infinite loop.
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5050,
        reloader_type="stat",
        use_reloader=True,
        extra_files=None,
    )

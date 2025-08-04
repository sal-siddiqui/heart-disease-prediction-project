from app import create_app

# create application
app = create_app()


if __name__ == "__main__":
    # initialize server
    app.run(host="localhost", port="5000", debug=True)

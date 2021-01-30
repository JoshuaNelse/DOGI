class Nav:
    BREED_CLASSIFIER = 'Dog Breed Classifier 🐕'
    METRICS = 'Data Analysis & Model Metrics 📈'
    ABOUT = 'About ❓'

    control = BREED_CLASSIFIER

    @staticmethod
    def get_nav_control():
        return Nav.control

    @staticmethod
    def set_nav_control(new_control):
        Nav.control = new_control

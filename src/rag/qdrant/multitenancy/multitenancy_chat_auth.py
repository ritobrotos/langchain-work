from multitenancy_constants import HISTORY_DEPARTMENT_NAME, SCIENCE_DEPARTMENT_NAME


def validate_credentials(entered_username, entered_password):
    users_db = [
        {'username': 'Emily', 'password': 'pass123', 'department': SCIENCE_DEPARTMENT_NAME},
        {'username': 'Benjamin', 'password': 'pass456', 'department': HISTORY_DEPARTMENT_NAME}
    ]

    # Iterate through the list of users to check if the entered username and password matches
    for user in users_db:
        if entered_username == user['username'] and entered_password == user['password']:
            print("User verified successfully")
            return user

    print("User not verified")
    return None
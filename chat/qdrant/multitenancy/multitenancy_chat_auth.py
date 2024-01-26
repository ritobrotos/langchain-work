def validate_credentials(entered_username, entered_password):
    users_db = [
        {'username': 'Emily', 'password': 'pass123', 'department': 'user_1'},
        {'username': 'Benjamin', 'password': 'pass456', 'department': 'user_2'}
    ]

    # Iterate through the list of users to check if the entered username and password matches
    for user in users_db:
        if entered_username == user['username'] and entered_password == user['password']:
            print("User verified successfully")
            return user

    print("User not verified")
    return None
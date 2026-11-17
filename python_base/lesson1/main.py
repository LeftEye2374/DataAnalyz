class User:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say_hello(self):
        print("Hello " + self.name)
    def print_age(self):
        print(self.age)

user1 = User("Misha", 22)

user1.say_hello()
user1.print_age()
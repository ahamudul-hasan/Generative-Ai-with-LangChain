from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Moonlight'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=5, description='A decimal value representing the cgpa of the student')

new_student = {'age': 32, 'email':'abc@gmail.com', 'cgpa': 3.25}

student = Student(**new_student)

print(student)

student_json = student.model_dump_json()
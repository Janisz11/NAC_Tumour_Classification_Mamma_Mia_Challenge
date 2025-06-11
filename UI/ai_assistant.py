import openai

api_key = ("sk-proj-6E8eaLEQ8WQei5fEJ19AFq7YBwslsiCAVDaXgKv0LksPL33zk5sfwKwg9HYTWSjFfVOYnsOwPf"
           "T3BlbkFJYZY_6BRHrQAjcQh4SeWYww8i-CRix45P8q75XyHMZAz2yQgW9_QyfqTM-ehKYISzpckygr0pUA")

client = openai.OpenAI(api_key=api_key)

models = client.models.list()

for model in models.data:
    print(model.id)

def send_message(self, event=None):
    user_text = self.user_input.get().strip()
    if not user_text:
        return

    self.chat_display.insert("end", f"Ty: {user_text}\n")
    self.user_input.delete(0, "end")

    self.assistant_messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.assistant_messages,
            temperature=0.2
        )
        assistant_reply = response.choices[0].message.content
        assistant_reply = response.choices[0].message.content
        self.chat_display.insert("end", f"Asystent: {assistant_reply}\n\n")
        self.chat_display.see("end")

        self.assistant_messages.append({"role": "assistant", "content": assistant_reply})

        if "load_patient_data" in assistant_reply:
            import re
            match = re.search(r"load_patient_data\((.*?)\)", assistant_reply)
            if match:
                patient_id = match.group(1).strip().strip("'\"")
                self.load_patient_data(patient_id)

    except Exception as e:
        self.chat_display.insert("end", f"Błąd: {str(e)}\n\n")


def load_patient_data(self, patient_id):
    self.chat_display.insert("end", f"[system] Załadowano dane pacjenta {patient_id}\n\n")
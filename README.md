# About the project

This software enables you to respond to job-related questions using your personal and professional information. It can be seamlessly integrated with external applications, such as bots, to automate responses in job application forms.



![Screenshot 2024-08-08 at 13 13 21](https://github.com/user-attachments/assets/d265d6c4-7a22-4b57-92ec-ccf242ed2273)


### How It Works?

The workflow is the following: 
1) First, a specialised classifier LLM tag the input question in order to know **which section/s from our CV we need to focus on.**
2) When the tags are defined, **we extract from our resume** the information of those sections and provide them to another LLM, in this case an expert QA for job-related question model.
3) With this information, this **model infers the answer of the input question based on the context** added in the prompt.

If you want to make your own testings, I provided a test directory where you can simulate template questions in LangSmith and see how accurated is the app for your use case. In my case, it retrieves more than 90% of accuracy :)


### About how to set up

Create Your Profile: **Go to the "resume" folder in the repository and create an info.md file**. Include the following headlines that summaries your professional profile. The better you detail them, the better responses the model will provide. (Avoid personal details like ID number, residencial address, at least, you are aware 100% about the data managment of the server where the LLM is running).

**Use the following sections exclusively:** 


- _ABOUT MY PROFILE_: It is like a brief about how you are, what you do, where you live, and general things regarding you...
- _TOOLS & SOFTWARE:_ Here you provide your skills or tools and years of experience with each one.
- _JOB PREFERENCES:_ Any preferencies like salary, remote/on-site preferences, part time/ full time and so on...
- _JOB EXPERIENCES_: A detail of our your work records and what did you do.
- _PROJECTS & USE CASES:_ Provide professional and academic projects (from job or personal ones).
- _EDUCATION_: Your background and education.
- _CERTIFICATIONS & COURSES_: Here provide a detail of your courses you made and also some certifications you may have.



### A glince of how it works in a demo:

I will also share some videos to **demonstrate how it works**: in this demo, I use gpt-4o-mini as router and llama 3 70b as qa bot.

Suppose I am applying for an AI Developer role. When we invoke the chain, we set the role and then in the input question in this case if I have experience with Python.
(This is hardcoded for explanation purposes, you can parameterize this later)
Look that the output also is parsed as an integer because the model detects that the question should be a number, not a string.

https://github.com/user-attachments/assets/504c08fb-01f0-4efc-b114-5966dc34873f


Also I tried with other topics and well this is the result:


https://github.com/user-attachments/assets/936777c7-0289-49ec-bb21-7bba44c60d83


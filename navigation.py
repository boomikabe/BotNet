import streamlit as st

import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename

import streamlit as st

import matplotlib.image as mpimg

import streamlit as st
import base64

import pandas as pd
import sqlite3

# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('11.jpg')


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    st.markdown(f'<h1 style="color:#8d1b92;text-align: center;font-size:30px;">{"Edge-Based Machine Learning for Immediate Botnet Detection and Response in IoT Networks"}</h1>', unsafe_allow_html=True)
    
    print()
    print()

    print()

    st.text("                 ")
    st.text("                 ")
    a = "  * To detect botnet attacks in real-time within IoT environments using hybrid machine learning algorithms and edge computing. * Implement/Hybrid Random Forest and Logistic Regression models to classify network traffic as either normal or indicative of a botnet attack. * Predictions are then stored in Boxcloud for further analysis and monitoring."
    
    st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:30px;font-family:Caveat, sans-serif;">{a}</h1>', unsafe_allow_html=True)

    st.text("                 ")
    st.text("                 ")
    
    st.text("                 ")
    st.text("                 ")
    



elif navigation()=='reg':
   
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Edge-Based Machine Learning for Immediate Botnet Detection and Response in IoT Networks"}</h1>', unsafe_allow_html=True)

    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:20px;">{"Register Here !!!"}</h1>', unsafe_allow_html=True)
    
    import streamlit as st
    import sqlite3
    import re
    
    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to check if a user already exists
    def user_exists(conn, email):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        if cur.fetchone():
            return True
        return False
    
    # Function to validate email
    def validate_email(email):
        pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        return re.match(pattern, email)
    
    # Function to validate phone number
    def validate_phone(phone):
        pattern = r'^[6-9]\d{9}$'
        return re.match(pattern, phone)
    
    # Main function
    def main():
        # st.title("User Registration")
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            # User input fields
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your name:</label>
                """,
                unsafe_allow_html=True
            )
            name = st.text_input("")
            
    
            # Create the text input field and password field
            # name = st.text_input("Your name")
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Password:</label>
                """,
                unsafe_allow_html=True
            )
            
            password = st.text_input("",type="password")
    
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Confirm Password:</label>
                """,
                unsafe_allow_html=True
            )
            
            confirm_password = st.text_input(" ",type="password")
            
            # ------
    
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Email ID:</label>
                """,
                unsafe_allow_html=True
            )
    
            email = st.text_input("  ")
            
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Phone Number:</label>
                """,
                unsafe_allow_html=True
            )
            
            
            phone = st.text_input("   ")
    
            col1, col2 = st.columns(2)
    
            with col1:
                    
                aa = st.button("REGISTER")
                
                if aa:
                    
                    if password == confirm_password:
                        if not user_exists(conn, email):
                            if validate_email(email) and validate_phone(phone):
                                user = (name, password, email, phone)
                                create_user(conn, user)
                                st.success("User registered successfully!")
                            else:
                                st.error("Invalid email or phone number!")
                        else:
                            st.error("User with this email already exists!")
                    else:
                        st.error("Passwords do not match!")
                    
                    conn.close()
                    # st.success('Successfully Registered !!!')
                # else:
                    
                    # st.write('Registeration Failed !!!')     
            
            with col2:
                    
                aa = st.button("LOGIN")
                
                if aa:
                    import subprocess
                    subprocess.run(['streamlit','run','Login.py'])
    
    
    
      
    if __name__ == '__main__':
        main()


if navigation() == "admin":
    
    
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Edge-Based Machine Learning for Immediate Botnet Detection and Response in IoT Networks"}</h1>', unsafe_allow_html=True)
     
  
        
    import subprocess
    
    subprocess.run(['streamlit','run','login.py'])    

    
if navigation() == "graphs":
    
    
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Edge-Based Machine Learning for Immediate Botnet Detection and Response in IoT Networks"}</h1>', unsafe_allow_html=True)
     
    st.image('pca.png')
    
    st.write("--------------------------------------------------------------------")
    
    
    st.image("graph.png")    
    
    st.write("--------------------------------------------------------------------")
   
    
    st.image("loss.png")    
    
    st.write("--------------------------------------------------------------------")
       
    
     
    # st.image("loss1.png")    
    
    # st.write("--------------------------------------------------------------------")
      
    
     
    st.image("com.png")    
    
    st.write("--------------------------------------------------------------------")
      
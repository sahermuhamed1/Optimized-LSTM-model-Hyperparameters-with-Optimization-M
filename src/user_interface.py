import streamlit as st
from PIL import Image, ImageDraw, ImageOps

def create_contact_section():
    """Create and display the contact information section."""
    st.markdown("---")
    st.header("Contact Information📩")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            # Open the image
            image = Image.open(r"...\profile.jpg")
            
            # Create a mask to make the image rounded
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, width, height), fill=255)
            
            # Apply the mask to the image
            image = ImageOps.fit(image, (width, height), method=0, bleed=0.0)
            image.putalpha(mask)
            
            # Display the image in Streamlit with the rounded shape
            st.image(image, width=150, caption="Profile Picture")
            
        except Exception as e:
            st.error("Error loading profile picture")
    
    with col2:
        st.markdown("""
        ### Connect with me
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/profile/)
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/profile)
        #### email: yourmail@gmail.com
        """)
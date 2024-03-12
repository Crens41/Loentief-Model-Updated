import streamlit as st
def app():
    st.title('Leontief Input_Output Model')
    col1,col2,col3= st.columns(3)
    with col1:
        st.image(image='loentief_model.PNG', width=700)
    st.write("Leontief model also known as the input-output analysis or The input-output model was named after the economist Wassily Leontief who pioneered its development in the 1930s.")
    st.markdown("")
    st.write("The input-output model is an economic tool is used to analyze interdependencies among different sectors within an economy. It quantifies the relationships between sectors by capturing the flows of goods, services, and payments between them. ")
    st.write("By representing these interactions in a matrix format, the input-output model enables economists and policymakers to understand the ripple effects of changes in one sector on others, assess economic impacts of various policies or shocks, and identify key drivers of economic activity and growth. ")
    st.markdown("")
    st.write("Overall, it provides valuable insights into the structure and dynamics of an economy's production and consumption patterns.")

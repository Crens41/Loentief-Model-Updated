def app():
    import streamlit as st
    from streamlit_option_menu import option_menu
    import input_output_2x2, input_output_3x3

    st.title("Leontief models")
    st.markdown("This model automatically solves a problem involving two industries and also a case of three industries ")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    option = st.selectbox(
        label= "Select the number of industries",
        options= ("Select","Two Industries", "Three Industries")
    )

    if option == "Two Industries":
        input_output_2x2.app()
    if option == "Three Industries":
        input_output_3x3.app()

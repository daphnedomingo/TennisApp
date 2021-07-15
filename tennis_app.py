import streamlit as st
import pandas as pd
import numpy as np
import pickle
import bz2

def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = pickle.load(data)
 return data

model = decompress_pickle('model.pbz2')

def predict_win(model, df):

    predictions_data = model.predict_proba(df)
    output= predictions_data[0][1] * 100
    return output

def main():
    st.title('Tennis Win Probability Prediction App')
    st.write('This is a web app to predict the win probability of a tennis player based on\
            several features that you can see below. Please adjust the value of each feature.\
            After that, click on the Predict button at the bottom to see the prediction.')

    tourney_list= pd.read_csv('tourney_list.csv')
    t_list= list(tourney_list['tourney_name'])
    Players_list= pd.read_csv('Players_list.csv')
    p_list= list(Players_list['Player'])
    round_list= ['F', 'SF', 'QF',  'R16', 'R32', 'R64','R128','BR', 'ER', 'RR']

# Making Sliders and Feature Variables
    form= st.form(key='myform')

    tourney_name = form.selectbox("Tournament",t_list)
    surface= tourney_list[tourney_list['tourney_name']==tourney_name]['surface'].values[0]
    draw_size=tourney_list[tourney_list['tourney_name']==tourney_name]['draw_size'].values[0]
    tourney_level= tourney_list[tourney_list['tourney_name']==tourney_name]['tourney_level'].values[0]
    best_of= tourney_list[tourney_list['tourney_name']==tourney_name]['best_of'].values[0]


    match_round= form.selectbox("Round",round_list)
    if best_of==3:
        num_sets = form.slider(label = 'Number of sets', min_value = 0,
                        max_value = 3,
                        value = 2,
                        step = 1)
    else:
        num_sets = form.slider(label = 'Number of sets', min_value = 0,
                            max_value = 6,
                            value = 4,
                            step = 1)
    mins_per_set= form.number_input(label= 'Average minutes per set', min_value=0,
                        max_value=None, value= 40)

    col1, col2 = form.beta_columns(2)

    with col1:
        col1.header('Player of Interest')
        Player1_name= col1.selectbox("Player Name",p_list)
        Player1_rank= col1.number_input(label= 'Player Rank', min_value=0,
                            max_value=None, value= 2000)

        Player1_Ace = col1.slider(label = 'Player Aces', min_value = 0,
                            max_value = 50,
                            value = 0,
                            step = 1)

        Player1_df = col1.slider(label = 'Player Double Faults', min_value = 0,
                            max_value = 50,
                            value = 0,
                            step = 1)

        Player1_1stServePct = col1.slider(label = 'Player 1st Serve Percentage', min_value = 0.0,
                            max_value = 100.0,
                            value = 50.0,
                            step = 0.5)

        Player1_1stServeWonPct = col1.slider(label = 'Player 1st Serve Won Percentage', min_value = 0.0,
                            max_value = 100.0 ,
                            value = 50.0,
                            step = 0.5)

        Player1_bpFaced = col1.slider(label = 'Player Break points Faced', min_value = 0,
                            max_value = 100 ,
                            value = 0,
                            step = 1)

        Player1_bpSavedPct= col1.slider(label = 'Player Break points Saved Percentage', min_value = 0.0,
                            max_value = 100.0 ,
                            value = 50.0,
                            step = 0.5)

    with col2:
        col2.header('Opponent')

        Player2_name= col2.selectbox("Opponent Name",p_list)
        Player2_rank= col2.number_input(label= 'Opponent Rank', min_value=0,
                            max_value=None, value= 2000)

        Player2_Ace = col2.slider(label = 'Opponent Aces', min_value = 0,
                            max_value = 100 ,
                            value = 0,
                            step = 1)

        Player2_df = col2.slider(label = 'Opponent Double Faults', min_value = 0,
                            max_value = 50 ,
                            value = 0,
                            step = 1)

        Player2_1stServePct = col2.slider(label = 'Opponent 1st Serve Percentage', min_value = 0.0,
                            max_value = 100.0 ,
                            value = 50.0,
                            step = 0.5)

        Player2_1stServeWonPct = col2.slider(label = 'Opponent 1st Serve Won Percentage', min_value = 0.0,
                            max_value = 100.0 ,
                            value = 50.0,
                            step = 0.5)

        Player2_bpFaced = col2.slider(label = 'Opponent Break points Faced', min_value = 0,
                            max_value = 100 ,
                            value = 0,
                            step = 1)

        Player2_bpSavedPct = col2.slider(label = 'Opponent Break points Saved Percentage', min_value = 0.0,
                            max_value = 100.0 ,
                            value = 50.0,
                            step = 0.5)

    p1_hand=Players_list[Players_list['Player']==Player1_name]['Hand'].values[0]
    p1_ht=Players_list[Players_list['Player']==Player1_name]['Height'].values[0]
    p1_ioc=Players_list[Players_list['Player']==Player1_name]['IOC'].values[0]
    p1_age=Players_list[Players_list['Player']==Player1_name]['AGE'].values[0]
    p2_hand=Players_list[Players_list['Player']==Player2_name]['Hand'].values[0]
    p2_ht=Players_list[Players_list['Player']==Player2_name]['Height'].values[0]
    p2_ioc=Players_list[Players_list['Player']==Player2_name]['IOC'].values[0]
    p2_age=Players_list[Players_list['Player']==Player2_name]['AGE'].values[0]


    # Mapping Feature Labels with Slider Values

    features = {
        'tourney_name': tourney_name,
        'surface':surface,
        'draw_size': draw_size,
        'tourney_level': tourney_level,
        'best_of': best_of,
        'round': match_round,
        'num_sets': num_sets,
        'mins_per_set': mins_per_set,
        'player1': Player1_name,
        '1_hand': p1_hand,
        '1_ht': p1_ht,
        '1_ioc': p1_ioc,
        '1_age':p1_age,
        '1_ace':Player1_Ace,
        '1_df':Player1_df,
        '1_1stSvPct':Player1_1stServePct,
        '1_1stSvWonPct':Player1_1stServeWonPct,
        '1_bp_faced':Player1_bpFaced,
        '1_bpSavedPct':Player1_bpSavedPct,
        '1_rank': Player1_rank,
        'player2': Player2_name,
        '2_hand': p2_hand,
        '2_ht': p2_ht,
        '2_ioc': p2_ioc,
        '2_age':p2_age,
        '2_ace':Player2_Ace,
        '2_df':Player2_df,
        '2_1stSvPct':Player2_1stServePct,
        '2_1stSvWonPct':Player2_1stServeWonPct,
        '2_bpfaced':Player2_bpFaced,
        '2_bpSavedPct':Player2_bpSavedPct,
        '2_rank': Player2_rank
    }


    # Converting Features into DataFrame

    features_df  = pd.DataFrame([features])

    # st.table(features_df)

    # Predicting Star Rating
    cb1, cb2, cb3, cb4, cb5 = form.beta_columns(5)

    submit = form.form_submit_button('Predict Win Probability')

    if submit:
        # st.table(features_df)
        # prediction=1
        prediction= predict_win(model, features_df)
        st.write(' Based on feature values, the win probability of '+ Player1_name +" for this match is " + str(int(prediction)) + "%")



if __name__ == '__main__':
	main()

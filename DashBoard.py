import dash
from dash import html
import torch 
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dash import Dash, html
import base64
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import io
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
from dash import no_update
import pandas as pd
import plotly.graph_objects as go
import numpy as np 

load_figure_template("LUX")

dir = "C:/Users/Luca.Dipalma/OneDrive - Bracco Imaging SPA/Desktop/GAN/ModelloSalvato/Modelli_epoca_1470"
os.chdir(dir)

# Step per plottare la prima immagine
def SelectImage():
    n = random.randint(0,100)
    print(n)
    if n>50:
        folder = "/Reali"
        risposta_corretta = 0
    else:
        folder = "/Fake"
        risposta_corretta = 1

    lista_immagini = os.listdir(dir + folder)
    num_images = len(lista_immagini)
    random_image = random.randint(0, num_images-1)
    #print("Numero immagini presenti: {}, numero estratto: {}".format(num_images, random_image))
    image_path = lista_immagini[random_image]
    image_path = dir + folder+"/"+image_path
    return image_path, risposta_corretta

def MostraEsempioReale(n_clicks):
    folder = "/RealiEsempio"
    lista_immagini = os.listdir(dir + folder)
    image_path = lista_immagini[n_clicks]
    image_path = dir + folder+"/"+image_path
    return image_path

def CreaTorta():
    df = pd.read_excel("{}/assets/salvadati.xlsx".format(dir))
    risposte_date = df['RisposteDate']
    risposte_corrette = df['RisposteCorrette']

    print(risposte_date)
    print(risposte_corrette)

    tot_risposte_date = []
    for risp in risposte_date:
        tot_risposte_date.append(eval(risp))

    tot_risposte_date = [item for sublist in tot_risposte_date for item in sublist]

    tot_risposte_corrette = []
    for risp in risposte_corrette:
        tot_risposte_corrette.append(eval(risp))

    tot_risposte_corrette = [item for sublist in tot_risposte_corrette for item in sublist]

    tot_risposte_date = np.array(tot_risposte_date)
    tot_risposte_corrette = np.array(tot_risposte_corrette)

    numero_reali_predette_reali = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==0]==0)
    numero_reali_predette_fake = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==0]==1)
    numero_fake_predette_fake = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==1]==1)
    numero_fake_predette_reali = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==1]==0)

    labels = ['Real considered real', 'Real considered fake', 'Fake considered fake', 'Fake considered real']
    values = [numero_reali_predette_reali, numero_reali_predette_fake, numero_fake_predette_fake, numero_fake_predette_reali]

    layout = go.Layout(#width=400, 
                       #height=350, 
                        #title='Overall results',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=go.layout.Margin(
                        l=0, #left margin
                        r=0, #right margin
                        b=0, #bottom margin
                        t=0) #top margin
                       ) #top margin
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)], layout=layout)
    fig.update_traces(textinfo='value')
    return fig

def VisualizzaSoloMieiDati(tot_risposte_date, tot_risposte_corrette):
    
    tot_risposte_date = np.array(tot_risposte_date)
    tot_risposte_corrette = np.array(tot_risposte_corrette)

    numero_reali_predette_reali = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==0]==0)
    numero_reali_predette_fake = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==0]==1)
    numero_fake_predette_fake = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==1]==1)
    numero_fake_predette_reali = np.count_nonzero(tot_risposte_date[tot_risposte_corrette==1]==0)

    print("numero_reali_predette_reali: {}".format(numero_reali_predette_reali))
    print("numero_reali_predette_fake: {}".format(numero_reali_predette_fake))
    print("numero_fake_predette_fake: {}".format(numero_fake_predette_fake))
    print("numero_fake_predette_reali: {}".format(numero_fake_predette_reali))

    labels = ['Real considered real', 'Real considered fake', 'Fake considered fake', 'Fake considered real']
    values = [numero_reali_predette_reali, numero_reali_predette_fake, numero_fake_predette_fake, numero_fake_predette_reali]

    layout = go.Layout(#width=400, 
                       #height=350, 
                        #title='Overall results',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=go.layout.Margin(
                        l=0, #left margin
                        r=0, #right margin
                        b=0, #bottom margin
                        t=0) #top margin
                       ) #top margin
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)], layout=layout)
    fig.update_traces(textinfo='value')
    return fig


# df = pd.DataFrame(columns=['RisposteDate','RisposteCorrette'])
# df.to_excel("{}/assets/salvadati.xlsx".format(dir))


img_shape = 128
correct_answer = []
given_answer = []
image_path = MostraEsempioReale(0)
pil_img = Image.open(image_path)
pil_img = pil_img.resize((img_shape,img_shape))

img_path_test, risposta_corretta = SelectImage()
correct_answer.append(risposta_corretta)
pil_img_test = Image.open(img_path_test)
pil_img_test = pil_img.resize((img_shape,img_shape))


numero_domande = 0
#numero_domande = 5


# img = Image.open('C:/Users/Luca.Dipalma/OneDrive - Bracco Imaging SPA/Desktop/GAN/ModelloSalvato/Modelli_epoca_1470/Fake/Generata_1.png')
# img_arr = np.array(img)
# # Create figure data with image trace
# fig = go.Figure()
# fig.add_trace(go.Image(z=img_arr))

# # Using base64 encoding and decoding
# def b64_image(image_filename):
#     with open(image_filename, 'rb') as f:
#         image = f.read()
#     return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

#app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = DashProxy(prevent_initial_callbacks=False, transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

GANText = "GAN stands for Generative Adversarial Network, which is a type of deep learning model that consists of two neural networks that work together to generate new data that is similar to a given training dataset. One network is called the generator, which creates new data samples, and the other is called the discriminator, which evaluates the authenticity of the generated data. The two networks are trained in an adversarial manner, meaning that the generator tries to create data that can fool the discriminator, while the discriminator tries to distinguish between real and fake data. Through this iterative process, the generator learns to create increasingly realistic data, and the discriminator learns to become more accurate in identifying real and fake data. GANs have been used in a variety of applications, including image and video generation, music synthesis, and text generation. They have also been used for data augmentation, which involves creating new training data from existing data to improve the performance of machine learning models."
SIDEBAR_STYLE = {
   "align-items": "center",
   "justify-content": "center",
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "30%",
    "height": "100vh",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll",
    "flex-direction": "column",
    "order": 0,
    #"display": "flex"
}

sidebar = html.Div(id="sidebar", children=
    [
        html.H3("Generative Adversarial Network", className="display-4", ),
        html.Hr(),
        html.P("What is a GAN?", className="lead"),
        html.P(GANText, className="lead"),
    ],
    style=SIDEBAR_STYLE,
)

test_iniziato = False  
sceltaFatta = False

app.layout = html.Div(id = 'parent', children = [
        dcc.Location(id='url', refresh=True),
        sidebar,
        html.Div(id='destra', children=[     
                html.Div([html.H1('AI Chest X-Ray GAN', style={'text-align': 'center'}),], className="first_div"),
                html.Button('Start test', id='start_test', className='start_test_div'),

                html.Div([
                    html.Div([
                        html.H5("Now I will show you 10 examples of real images.", className= "scrittaEsempi"),
                        html.Img(id='image_example', src=pil_img, className='immagine_esempio'),
                        html.Button("Next image", id='bottone_examples', className = "nextImage"),
                        #html.Button("Next image", id='bottone_examples', children=[html.Img(src=app.get_asset_url('freccia.png'), className="immagine_freccia")], className = "nextImage"),
                        html.Button("Skip tutorial", id='skip_im', className = "skipButton"),                 
                        ], className='esempiImmagine'),      
                ], className='blocco_esempi', id='esempi_reali', style={'display':'none'}),


                html.Div([
                    html.Div([
                        html.H5("How many images do you want to classify?", style={'textAlign': 'center','margin-bottom': '5%', 'padding-top':"2%"}),
                        dcc.Dropdown([10, 15, 20], id='scelta_numero_domande', placeholder="Give me an answer...", style={"width":"50vh",  "align-items": "center", "justify-content": "center"}),
                        html.Div(id='dd-output-container')
                        ], className='content-inner')
                ], className='content', id='div_scelta_numero', style={'display':'none'}),

            html.Div(children=[
                #dcc.Graph(figure=fig, config={'displayModeBar': False}), # Always display the modebar)
                
                html.Div(children=[
                    html.Img(id='image', src=pil_img_test, className='center', style={'text-align': 'center'}),
                ], className="container_immagine", id="cont_imm"),
                
                html.Div(children=[
                    html.Div(children=[
                        html.H5("Is this image real or fake?"),
                    ], className="domanda_div"),
                    html.Div(children=[
                        dcc.RadioItems(
                        [
                            {
                                "label": "Real", #html.Div(
                                #     [
                                #         html.H1("Real", style={'font-size': 15, 'padding-right': 10}),
                                #         html.Img(src="assets/x-ray.svg", height=30),
                                #     ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                                # ),
                                "value": 0,
                            },
                            {
                                "label": "Fake", #html.Div(
                                #     [
                                #         html.H1("Fake", style={'font-size': 15, 'padding-right': 10}),
                                #         html.Img(src="assets/artificial-intelligence.svg", height=30),
                                #     ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
                                # ),
                                "value": 1,
                            }
                        ], labelStyle={'display': 'block', 'padding-righ':'50px'}, id="radio-button", className="dash-radioitems", style={'margin-top': '5%'}),
                    ]),
                    html.Button('Go!', id='submit-button', n_clicks=0, className="button-85",  style={'text-align': 'center', 'margin-top': '8%'}),
                ], className="parte_questionario", id="parte_quest"),
                html.Div(className="scritta_sotto", id='output'),         
            ], className='content_2', id="div_blocco_immagine", style={'display': 'none'}),

            html.Div(children=[
                html.A(html.Button('Try again!', id='try_ag', className='try_again'), href='/'),
                html.Div(dcc.Graph(id="graficoATorta", className="grafico_a_torta_totale",  responsive=True), className="divContainerGrafico"),
                ], className="fine_test", id="fine_test_div", style={'display': 'none'}),
        ], className="parte_Destra")
], 
className='image-container')



# Si attiva quando inizi il test
@app.callback(
        Output('start_test', 'style'),
        Output('esempi_reali', 'style'),
        [Input('start_test', 'n_clicks')]
)
def activate_div(n_clicks):
    global test_iniziato
    if n_clicks is None:
        test_iniziato = False
        return {'display': 'block'}, {'display': 'none'}
    else:
        test_iniziato = True
        return {'display': 'none'}, {'display': 'block'}
        

# Visualizza prossimo esempio
@app.callback(
    Output('image_example', 'src'),
    Output('esempi_reali', 'style'),
    Output('div_scelta_numero', 'style'),
    [Input('bottone_examples', 'n_clicks')]
)
def activate_div(n_clicks):
    global test_iniziato
    if not test_iniziato:
        print(test_iniziato)
        return no_update
    
    if n_clicks is None:
        img_path = MostraEsempioReale(0)
        print(img_path)
        new_pil_img = Image.open(img_path)
        new_pil_img = new_pil_img.resize((img_shape,img_shape))
        return new_pil_img, {'display':'block'}, {'display':'none'}, 
    else:
        if(n_clicks<10):
            img_path = MostraEsempioReale(n_clicks)
            new_pil_img = Image.open(img_path)
            new_pil_img = new_pil_img.resize((img_shape,img_shape))
            return new_pil_img, {'display':'block'}, {'display':'none'}, 
        else:
            return "", {'display':'none'}, {'display':'block'}, 


# Salta tutorial
@app.callback(
    Output('esempi_reali', 'style'),
    Output('div_scelta_numero', 'style'),
    [Input('skip_im', 'n_clicks')]
)
def activate_div(n_clicks):
    if not test_iniziato:
        return no_update
    
    if n_clicks is None:
        return no_update
    else:
        return {'display':'none'}, {'display':'block'}, 
        
    
# Si attiva quando classifichi una domanda
@app.callback(
    Output('output', 'children'),
    Output('image', 'src'),
    Output(component_id='parte_quest', component_property='style'),
    Output(component_id='cont_imm', component_property='style'),
    Output(component_id='fine_test_div', component_property='style'),
    Output(component_id='graficoATorta', component_property="figure"),
    [Input('submit-button', 'n_clicks')],
    [State('radio-button', 'value')])
def update_output(n_clicks, value):

    global risposta_corretta 
    global correct_answer
    global given_answer
    global test_iniziato
    global sceltaFatta

    if not test_iniziato:
        return no_update
    
    if not sceltaFatta:
        return no_update

    if n_clicks > 0 and value is not None:

        given_answer.append(value)
        img_path, ris_corr = SelectImage()

        graficoDaPlottare = go.Figure()

        if(len(given_answer)<numero_domande):
            risposta_corretta = ris_corr
            correct_answer.append(risposta_corretta)
            new_pil_img = Image.open(img_path)
            new_pil_img = new_pil_img.resize((img_shape,img_shape))
            return "This is the question {}/{}.".format(len(given_answer)+1, numero_domande), new_pil_img, {'visibility': 'visible'}, {'visibility': 'visible'}, {'display':'none'}, graficoDaPlottare
        else:
            # Il test è finito
            a = np.array(correct_answer)
            b = np.array(given_answer)
            #print("Risposte giuste: {}\nRisposte date: {}".format(a,b))
            numero_corrette = np.count_nonzero(a==b)
            df = pd.read_excel("{}/assets/salvadati.xlsx".format(dir))
            new_row = {'RisposteDate': given_answer, 'RisposteCorrette': correct_answer}
            df = pd.concat([df, pd.DataFrame([new_row])])
            #df.to_excel("{}/assets/salvadati.xlsx".format(dir))
            graficoDaPlottare = VisualizzaSoloMieiDati(given_answer, correct_answer) #CreaTorta()
            numero_risposte_date = len(given_answer)
            correct_answer = []
            given_answer = []

            return "You obtained a score of {}/{}!".format(numero_corrette,numero_risposte_date), '', {'visibility': 'hidden'}, {'visibility': 'hidden'}, {'display':'block'},  graficoDaPlottare
    else:
        return "This is the question {}/{}.".format(len(given_answer)+1, numero_domande), '', {'visibility': 'visible'}, {'visibility': 'visible'}, {'display':'none'}, graficoDaPlottare


# Si attiva quando rispondi a quante immagini vuoi classificare
@app.callback(
    Output('output', 'children'),
    Output(component_id='div_scelta_numero', component_property='style'),
    Output(component_id='div_blocco_immagine', component_property='style'),
    Input('scelta_numero_domande', 'value')
)
def update_output(value):
    global correct_answer
    global given_answer
    global numero_domande
    global sceltaFatta

    if not test_iniziato:
        return no_update
    if sceltaFatta:
        return no_update

    if value is not None:
        correct_answer = []
        given_answer = []
        image_path, risposta_corretta = SelectImage()
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((img_shape,img_shape))
        correct_answer.append(risposta_corretta)
        numero_domande = value
        #print(numero_domande)
        sceltaFatta = True
        #print("qui dentroooo")
        return "This is the question {}/{}.".format(len(given_answer)+1, numero_domande), {'display': 'none'}, {'display': 'block'}
        #return {'visibility': 'hidden'}, {'visibility': 'visible'} #, f'You have selected {value}'
    else:
        #print("quaaaa")
        return "This is the question {}/{}.".format(len(given_answer)+1, numero_domande), {'display': 'block'}, {'display': 'none'}
        #return {'visibility': 'visible'}, {'visibility': 'hidden'} #, 'You have not selected yet'


# Si attiva quando refreshi la pagina
@app.callback(
    Output('output', 'children'),
    Input('url', 'pathname'),
    State('output', 'children')
)
def reset_global_variables(pathname, current_value):
    # If the page is refreshed or the URL changes, reset the global variable
    global sceltaFatta
    global test_iniziato
    sceltaFatta = False
    test_iniziato = False  
    
    # Your callback code goes here
    return current_value


if __name__ == '__main__': 
    app.run_server(debug=False)#, dev_tools_hot_reload = False)






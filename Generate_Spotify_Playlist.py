import spotipy
from spotipy.oauth2 import SpotifyOAuth
from textblob import TextBlob
import webbrowser
import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import logging
import tkinter as tk
from tkinter import messagebox, ttk
import wave

# Configurações de autenticação
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="e2b6c40e592a452489fd08e46ae7c39d",
                                               client_secret="a68529b608354d078e5591f07f5cb19e",
                                               redirect_uri="http://localhost:8888/callback",
                                               scope="playlist-modify-public user-top-read"))

# Configuração do logger
log_filename = r"C:\Users\fagne\OneDrive\[[[ARTICULAR\[[[AI-FILES\execution_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Variáveis globais
audio_filename_wav = "audio_recording.wav"
recording = False
fs = 16000
duration = 10  # Duração da gravação em segundos
selected_microphone = None
audio = None

# Função para obter as músicas mais tocadas de um gênero específico
def get_top_songs_by_genre(genre):
    logging.info(f"Buscando músicas do gênero: {genre}")
    results = sp.search(q=f'genre:{genre}', type='track', limit=50)
    songs = [(track['uri'], track['name'], track['artists'][0]['name']) for track in results['tracks']['items']]
    return songs

# Função para analisar o input do usuário
def analyze_input(user_input):
    logging.info("Analisando o input do usuário...")
    blob = TextBlob(user_input)
    logging.info(f"Análise do input: {blob}")
    return str(blob)

# Função para criar uma playlist e adicionar músicas
def create_playlist(user_input):
    logging.info("Criando playlist com base no input do usuário...")
    analyzed_input = analyze_input(user_input)
    genres = analyzed_input.lower().split()
    
    # Obter as músicas mais tocadas dos gêneros especificados
    songs = []
    for genre in genres:
        songs.extend(get_top_songs_by_genre(genre))
    
    # Verificar se há músicas suficientes
    if not songs:
        logging.info("Nenhuma música encontrada para os gêneros especificados.")
        return "", ""  # Retorna valores padrão em vez de None
    
    # Criar uma nova playlist
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user_id, f"Sugestões de {' e '.join(genres).capitalize()}", public=True, description=f"Playlist criada com base nos gêneros {', '.join(genres)}.")
    
    # Adicionar músicas à playlist
    song_uris = [song[0] for song in songs[:10]]
    sp.playlist_add_items(playlist['id'], song_uris)
    
    # Listar as músicas selecionadas
    logging.info("Playlist criada com sucesso! Confira no seu Spotify.")
    logging.info("Músicas adicionadas:")
    for i, song in enumerate(songs[:10], 1):
        logging.info(f"{i}. {song[1]} - {song[2]}")
    
    # Retornar o link da playlist
    playlist_url = playlist['external_urls']['spotify']
    playlist_uri = playlist['uri']
    logging.info(f"Acesse sua playlist aqui: {playlist_url}")
    return playlist_url, playlist_uri

# Função para abrir a playlist no aplicativo do Spotify
def open_playlist_in_spotify(playlist_uri):
    logging.info("Abrindo a playlist no aplicativo do Spotify...")
    webbrowser.open(f"spotify:playlist:{playlist_uri.split(':')[-1]}")

# Função para iniciar a gravação de áudio
def record_audio():
    global recording, rec_button, root, selected_microphone, audio

    if recording:
        return

    recording = True
    logging.info("Gravando áudio...")
    rec_button.config(text="Gravando...", bg="green", fg="white")

    # Configurar o dispositivo de entrada de áudio selecionado pelo usuário
    sd.default.device = (selected_microphone, None)

    # Verificar o número de canais suportados pelo dispositivo selecionado
    device_info = sd.query_devices(selected_microphone)
    channels = min(device_info['max_input_channels'], 1)

    # Iniciar a gravação
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')

    def finish_recording():
        global recording, audio
        recording = False
        
        sd.wait()  # Espera a gravação terminar
        
        # Salvar o áudio em um arquivo WAV e garantir que o arquivo seja fechado corretamente
        write(audio_filename_wav, fs, audio)
        
        logging.info("Gravação concluída!")
        
        # Converter áudio em texto e criar a playlist automaticamente após a gravação
        start_process()

    # Agendar finish_recording após a duração usando root.after
    root.after(duration * 1000, finish_recording)

# Função para converter áudio em texto
def speech_to_text(filename):
    logging.info("Convertendo áudio em texto...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    with wave.open(filename, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalizar o áudio

    # Processar áudio
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    logging.info(f"Transcrição concluída: {transcription}")
    
    return transcription

# Função para iniciar o processo de criação da playlist
def start_process():
    messagebox.showinfo("Processamento", "O áudio está sendo processado. Por favor, aguarde...")
    
    # Converter áudio em texto
    user_input = speech_to_text(audio_filename_wav)
    
    logging.info(f"Texto reconhecido: {user_input}")

    # Criar playlist com base no texto reconhecido
    playlist_link, playlist_uri = create_playlist(user_input)
    
    if playlist_link:
        logging.info(f"Link da playlist: {playlist_link}")
        
        # Abrir a playlist no aplicativo do Spotify
        open_playlist_in_spotify(playlist_uri)
        
        # Abrir a playlist no navegador padrão
        webbrowser.open_new_tab(playlist_link)

# Função para selecionar o microfone
def select_microphone():
    global selected_microphone, root
    try:
        selected_microphone = microphone_listbox.curselection()[0]
        root.destroy()
        show_rec_button()
    except IndexError:
        messagebox.showerror("Erro", "Por favor, selecione um microfone.")

# Função para mostrar a lista de microfones
def show_microphone_list():
    global root, microphone_listbox
    root = tk.Tk()
    root.title("Selecione o Microfone")

    tk.Label(root, text="Selecione o microfone para gravação:").pack(pady=10)
    
    microphone_listbox = tk.Listbox(root, height=15, width=50)  # Aumentar o tamanho da Listbox
    microphone_listbox.pack(pady=10)
    
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            microphone_listbox.insert(tk.END, device['name'])
    
    tk.Button(root, text="Selecionar", command=select_microphone).pack(pady=10)
    
    root.mainloop()

# Função para mostrar o botão REC
def show_rec_button():
    global root, rec_button
    root = tk.Tk()
    root.title("Gravador de Áudio")

    rec_button = tk.Button(root, text="REC", bg='red', fg='white', font=('Helvetica', 12, 'bold'), command=record_audio, height=2, width=10)
    rec_button.pack(pady=20)

    root.mainloop()

# Mostrar a lista de microfones ao iniciar o script
show_microphone_list()

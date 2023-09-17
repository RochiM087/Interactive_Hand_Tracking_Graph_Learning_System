import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Função para desenhar um gráfico em tempo real
def draw_graph(x, y):
    plt.clf()  # Limpa o gráfico anterior
    plt.plot(x, y, '-')  # Plota os pontos x e y como linhas
    plt.xlabel('Tempo')
    plt.ylabel('Posição')
    plt.title('Posição em função do tempo')
    plt.pause(0.001)  # Pausa para atualizar o gráfico

# Função para realizar o tracking do ponto da mão e do carrinho
def track_hand_and_car():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)  # Inicializa a câmera
    
    # Carrega a imagem do carrinho e da pista
    car_image = cv2.imread('nave.png')
    track_image = cv2.imread('espaco.png')

    # Inicializa o detector de mãos do MediaPipe
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        x_values = []
        y_values = []
        time_values = []

        history_points = []

        reset_graph = False

        while cap.isOpened():
            success, image = cap.read()  # Lê um frame da câmera

            if not success:
                print("Erro ao ler o frame")
                break

            # Redimensiona a imagem do carrinho para um tamanho adequado
            car_image_resized = cv2.resize(car_image, (50, 50))

            # Redimensiona a imagem da pista para o tamanho da imagem da câmera
            track_image_resized = cv2.resize(track_image, (image.shape[1], image.shape[0]))

            # Sobrepondo a imagem da pista na imagem da câmera
            image = cv2.addWeighted(image, 0.7, track_image_resized, 0.3, 0)

            # Processa a imagem com o detector de mãos do MediaPipe
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                hand_landmarks = results.multi_hand_landmarks[0]  # Seleciona a primeira mão detectada
                # Obtém as coordenadas normalizadas do ponto da mão que você deseja rastrear
                x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

                # Converte as coordenadas normalizadas para as coordenadas da imagem
                image_height, image_width, _ = image.shape
                x_px = int(x * image_width)

                # Define o valor do eixo Y no meio da imagem
                y_px = int(image_height / 2)

                # Verifica se a tecla "R" foi pressionada para redefinir o gráfico
                if reset_graph:
                    x_values = []
                    y_values = []
                    time_values = []
                    history_points = []
                    reset_graph = False

                # Armazena as coordenadas e o tempo em listas
                x_values.append(x_px)
                time_values.append(len(time_values))

                # Desenha um círculo no ponto rastreado
                cv2.circle(image, (x_px, y_px), 5, (0, 255, 0), -1)

                # Adiciona as coordenadas atuais à lista de histórico
                history_points.append((x_px, y_px))

                # Traça o histórico de pontos na imagem
                for point in history_points:
                    cv2.circle(image, point, 2, (0, 255, 0), -1)

                # Sobrepondo a imagem do carrinho na palma da mão
                x_offset = x_px - int(car_image_resized.shape[1] / 2)
                y_offset = y_px - int(car_image_resized.shape[0] / 2)
                image[y_offset:y_offset + car_image_resized.shape[0], x_offset:x_offset + car_image_resized.shape[1]] = car_image_resized

            # Mostra a imagem com o ponto rastreado e o traçado
            cv2.imshow('Hand Tracking', image)

            # Desenha o gráfico de posição em função do tempo
            draw_graph(time_values, x_values)

            # Verifica se a tecla "R" foi pressionada para redefinir o gráfico
            key = cv2.waitKey(1)
            if key & 0xFF == ord('r'):
                reset_graph = True

            # Verifica se a tecla "Q" foi pressionada para encerrar o processo
            if key & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Chama a função para realizar o tracking do ponto da mão e do carrinho
track_hand_and_car()

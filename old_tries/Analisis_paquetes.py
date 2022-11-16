import signal
from scapy.all import *
from scapy.layers.dot11 import Dot11, Dot11Elt, PrismHeader
from threading import Thread
import pandas
import time
import os
import sys

# Guardaremos los datos recibidos en un DataFrame con la MAC origen, nombre, nivel de señal y canal.
probeRequestFrames = pandas.DataFrame(columns=["ID", "SRCMAC", "Timestamp", "SSID", "dBm_Signal", "Channel"])
# Configuramos la MAC origen como índice del DataFrame.
probeRequestFrames.set_index("ID", inplace=True)



def callback(packet):
    packet_id = 0
    if not packet.haslayer(Dot11):
        return
    # Las tramas Probe Request tienen el subtipo 0x0004. Además aseguramos que estén dirigidos a la dirección
    # broadcast.
    if packet.type == 0 and packet.subtype == 4:        
        # Obtenemos la MAC origen.
        srcmac = packet.addr2
        # El nombre del dispositivo.
        ssid = packet[Dot11Elt].info.decode()
        # Intentamos capturar el nivel de señal.
        try:
            dbm_signal = packet.dBm_AntSignal
        except:
            dbm_signal = "N/A"
        # El canal donde se ha emitido la trama.
        try:
            channel = packet[PrismHeader].channel
        except:
            channel = "?"
        timestamp = time.asctime()
        if probeRequestFrames.empty:
            probeRequestFrames.loc[packet_id] = (srcmac, timestamp, ssid, dbm_signal, channel)
        else:
            packet_id = probeRequestFrames.shape[0]
            probeRequestFrames.loc[packet_id] = (srcmac, timestamp, ssid, dbm_signal, channel)          
        probeRequestFrames.to_csv('ProbeRequest.csv')


# Función que imprime las tramas capturadas.
def print_all():
    while True:
        os.system("clear")
        print(probeRequestFrames)
        time.sleep(5)

# Función que cambiará el puerto de escucha cada 0.5 segundos.
def change_channel():
    ch = 1
    while True:
        os.system(f"sudo iwconfig {interface} channel {ch}")
        ch = ch % 14 + 1
        time.sleep(0.5)

# Funcion que esperará hasta que el usuario quiera terminar la ejecución (CTRL+C).
def signal_handler(signal, frame):
    print('\n=====================')
    print('Execution aborted by user.')
    print('=====================')
    os.system('kill -9 ' + str(os.getpid()))
    sys.exit(1)

# Main donde se ejecutará el script. Si la interfaz usada no se llama "wlan0mon", cambiar el valor de la variable
# "interface". Se dedicará un hilo para para imprimir y para cambiar de canal.
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # Nombre de la interfaz.
    interface = "wlan0mon"
    # Hilo dedicado a imprimir las tramas capturadas.
    printer = Thread(target=print_all)
    printer.daemon = True
    printer.start()
    # Hilo dedicado a cambiar el canal de escucha.
    channel_changer = Thread(target=change_channel)
    channel_changer.daemon = True
    channel_changer.start()
    # Comienzo del sniff.
    sniff(prn=callback, iface=interface)

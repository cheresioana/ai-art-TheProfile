import os
import datetime
import paramiko
import json

background_file = 'doc/background.jpg'
dataset_folder = './doc/mini_dataset8'



def set_connection():

    with open("credentials.json") as json_data_file:
        data = json.load(json_data_file)
    print(data)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=data['hostname'], username=data['username'], password=data['passwd'], port=data['port'])
    return ssh_client


def download_latest_weights(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command("cat /home/adigro/cheres/training_checkpoints2/checkpoint")
    lines = stdout.readlines()
    lines = lines[0].split("\"")
    line = lines[1]
    print(line)
    stdin, stdout, stderr = ssh_client.exec_command("cd cheres/training_checkpoints2; ls " + line + "*")
    lines = stdout.readlines()
    ftp_client = ssh_client.open_sftp()
    ftp_client.get('/home/adigro/cheres/training_checkpoints2/checkpoint', './gan/training_checkpoints/checkpoint')
    for line in lines:
        print(line)
        line = line[:-1]
        ftp_client.get('/home/adigro/cheres/training_checkpoints2/' + line, './gan/training_checkpoints/' + line)

def upload_images(ssh_client):
    ftp_client = ssh_client.open_sftp()
    for item in os.listdir("dataset/train"):
        print(item)
        ftp_client.put('./dataset/train/' + item, '/home/adigro/cheres/train/' + item)
    ftp_client.close()

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    ssh_client = set_connection()
    print("Strated uploading images...")
    download_latest_weights(ssh_client)
    #upload_images(ssh_client)
    print("Timp de uploadare")
    '''print(datetime.datetime.now() - begin_time)
    #print("Started docker container...")
    #stdin, stdout, stderr = ssh_client.exec_command("docker run -v ~/cheres/training_checkpoints2:/home/cheres/training_checkpoints2 --gpus all -i cheres-gan2")
    #lines = stdout.readlines()
    #lines2 = stderr.readlines()
    #print(lines2)
    #print(lines)
    print("Timp train")
    print(datetime.datetime.now() - begin_time)
    print("Started downloading weights...")
    download_latest_weights(ssh_client)
    print("Timp de download weight-uri")
    print(datetime.datetime.now() - begin_time)'''
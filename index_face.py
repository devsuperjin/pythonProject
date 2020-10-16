import face_recognition
import numpy as np
import os
faces=[]
names=[]
def get_all_filename(path):
    return os.listdir(path)

def ctreate_face_embeddings(image_path):
    for i in get_all_filename(image_path):
        all_images = face_recognition.load_image_file(image_path+i)
        try:
            all_face_encoding = face_recognition.face_encodings(all_images)[0]
            faces.append(all_face_encoding)
            names.append(i.split('.')[0])
            #print(all_face_encoding)
        except IndexError:
            print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    embedding_faces = np.asarray(faces)
    np.save('./cgj_face.npy', embedding_faces)
    embedding_names = np.asarray(names)
    np.save('./cgj_name.npy', embedding_names)
if __name__ == '__main__':
    ctreate_face_embeddings('./xuehao/')

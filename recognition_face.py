import face_recognition
import numpy as np
def load_dataset(face_path,name_path):
    compare_emb=np.load(face_path)
    names_list=np.load(name_path)
    return compare_emb,names_list
def compare_embadding(compare_emb,unknow_iamge_path):
    unknown_image = face_recognition.load_image_file(unknow_iamge_path)
    try:
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        face_distances = face_recognition.face_distance(compare_emb, unknown_face_encoding)
        for i, face_distance in enumerate(face_distances):
            if face_distance < 0.3:
                print(i,names_list[i],face_distance)
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images.")
if __name__=='__main__':
    compare_emb,names_list=load_dataset('cgj_face.npy','cgj_name.npy')
    compare_embadding(compare_emb,'./xuehao/051216163428.jpg')
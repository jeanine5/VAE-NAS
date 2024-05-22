import numpy as np
import scipy


def load_smpl_take_file(in_path):
    try:
        data = np.load(in_path)
        print(list(data.keys()))
        return data['poses'], data['trans']
    except Exception as e:
        return None, None


def extract_joint_data(pose_data):
    # joint_data_size = len(self.joint_names) * 3
    joint_data = pose_data[:, :22 * 3]
    return joint_data


path = 'ACCAD/ACCAD/Male2Running_c3d/C3 - run_poses.npz'

smpl_data, root_positions = load_smpl_take_file(path)

joint_data = extract_joint_data(smpl_data)
joint_data = joint_data.reshape((joint_data.shape[0], joint_data.shape[1] // 3, 3))

quat_data = np.zeros((joint_data.shape[0], joint_data.shape[1], 4))

for frame in range(joint_data.shape[0]):
    for joint in range(joint_data.shape[1]):
        rot = scipy.spatial.transform.Rotation.from_rotvec(joint_data[frame, joint].tolist(), degrees=False)
        quat_data[frame, joint] = rot.as_quat()
# I think this is in x, y, z, w format
# this next line converts that to a numpy array with W, x, y, z ordering
# joint_value = np.array([q[3], q[0], q[1], q[2]])

print(quat_data.shape[1])

# Reshape the quat_data array
reshaped_data = quat_data.reshape(quat_data.shape[0], -1)

#
# # Save the reshaped data to a CSV file
csv_file_path = 'quat_data.csv'
np.savetxt(csv_file_path, reshaped_data, delimiter=',')

npy_file_path = 'quat_data.npy'
np.save(npy_file_path, reshaped_data)


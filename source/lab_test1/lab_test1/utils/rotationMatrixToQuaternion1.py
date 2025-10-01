# filename: rotationMatrixToQuaternion1.py
import numpy as np

def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """
    수치오차로 살짝 비틀린 R을 최근접 회전행렬로 보정.
    (Polar Decomposition via SVD)
    """
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    # det 음수면 마지막 축 반전해 올바른 회전행렬로 맞춤
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho

def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    단일 3x3 회전행렬 → 쿼터니언 [w,x,y,z] (Isaac 규약)
    참고: 수치 안정성을 위해 trace 분기 공식을 사용.
    """
    # 보정 및 안정화
    R = _orthonormalize(R)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    else:
        if (m00 > m11) and (m00 > m22):
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    # 정규화 (수치 안정)
    q /= np.linalg.norm(q) + 1e-12
    # 부호 통일(옵션): w를 양수로 맞추면 연속성에 도움
    if q[0] < 0:
        q = -q
    return q

def rotationMatrixToQuaternion1(R: np.ndarray) -> np.ndarray:
    """
    3x3 또는 (N,3,3) 회전행렬을 받아 쿼터니언 [w,x,y,z]로 변환.
    반환 shape: (4,) 또는 (N,4)
    """
    R = np.asarray(R)
    if R.ndim == 2:
        assert R.shape == (3, 3), "입력은 3x3 회전행렬이어야 합니다."
        return _mat_to_quat_wxyz(R)
    elif R.ndim == 3:
        assert R.shape[1:] == (3, 3), "배치 입력은 (N,3,3) 이어야 합니다."
        qs = np.stack([_mat_to_quat_wxyz(R[i]) for i in range(R.shape[0])], axis=0)
        return qs
    else:
        raise ValueError("입력 차원이 올바르지 않습니다. 3x3 또는 (N,3,3) 만 허용됩니다.")

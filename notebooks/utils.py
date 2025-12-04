import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np

class SphericalMeanShift:
    def __init__(self, kappa=50.0, max_iter=100, tol=1e-4, merge_angle_deg=5.0):
        
        self.kappa = kappa       # "узость" ядра: чем больше, тем локальнее
        self.max_iter = max_iter
        self.tol = tol          # критерий сходимости для одной траектории
        self.merge_angle_deg = merge_angle_deg    # порог склейки мод в градусах


    def l2_normalize_rows(self, X, eps=1e-12):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return X / norms


    def fit_predict(self, X):

        # 1) Нормируем входные эмбеддинги на сферу
        X_norm = self.l2_normalize_rows(X)
        n_samples, n_features = X_norm.shape

        # 2) Инициализируем траектории: начинаем из самих точек
        modes = X_norm.copy()

        for it in range(self.max_iter):
            # Косинусное сходство между текущими точками и всеми данными
            # modes: (n, d), X_norm: (n, d) -> (n, n)
            sim = modes @ X_norm.T

            # vMF-ядро: веса по exp(kappa * cos)
            weights = np.exp(self.kappa * sim)
            # Нормируем веса по строкам
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

            # Обновляем точки mean-shift шагом: взвешенное среднее всех X_norm
            new_modes = weights @ X_norm  # (n, d)
            new_modes = self.l2_normalize_rows(new_modes)

            # Максимальный сдвиг среди всех точек
            shifts = np.linalg.norm(new_modes - modes, axis=1)
            max_shift = shifts.max()

            modes = new_modes

            if max_shift < self.tol:
                # print(f"Mean-shift converged at iter {it}, max_shift={max_shift}")
                break

        # 3) Кластеризация мод: склеиваем близкие по углу
        # cos(angle) = u·v, angle = arccos(cos)
        merge_angle_rad = np.deg2rad(self.merge_angle_deg)
        cos_thresh = np.cos(merge_angle_rad)

        labels = -np.ones(n_samples, dtype=int)
        centers = []

        for i in range(n_samples):
            if labels[i] != -1:
                continue  # уже отнесён к какому-то кластеру

            # Новый кластер, стартуем с моды i
            center = modes[i]
            cluster_id = len(centers)

            # Находим все моды, достаточно близкие по углу к этой
            sims = modes @ center  # (n,)
            in_cluster = sims >= cos_thresh

            labels[in_cluster] = cluster_id
            # Можно обновить центр как среднее этих мод (но они и так близки)
            cluster_center = self.l2_normalize_rows(modes[in_cluster].mean(axis=0, keepdims=True))[0]
            centers.append(cluster_center)

        centers = np.vstack(centers) if centers else np.zeros((0, n_features))

        return labels, centers



def l2_normalize_rows(X, eps=1e-12):
    """
    Нормируем каждую строку до единичной L2-нормы (||x|| = 1).
    """
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def plot_clusters_3d_on_sphere(
    X,
    labels,
    center,
    title="Spherical k-means 3D",
    width=1200,
    height=900
):
    """
    Большой 3D-график кластеров на сфере (Plotly).

    width, height — размер графика в пикселях.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.shape[1] != 3:
        raise ValueError("X должен иметь форму (n_samples, 3)")

    # ---- сфера ----
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    sphere_surface = go.Surface(
        x=xs,
        y=ys,
        z=zs,
        opacity=0.15,
        showscale=False
    )

    # ---- точки ----
    points_trace = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=labels,
            colorscale="Viridis",
            opacity=0.85
        ),
        name="points"
    )

    traces = [sphere_surface, points_trace]

    # ---- центр ----
    if center:
                # глобальный центр
        global_mean_embed_3d = np.mean(X, axis=0)
    
        # ---- нормализация на сферу ----
        norm = np.linalg.norm(global_mean_embed_3d)
        if norm == 0:
            center_vec = global_mean_embed_3d
        else:
            center_vec = global_mean_embed_3d / norm
        
        centers_trace = go.Scatter3d(
            x=[center_vec[0]],
            y=[center_vec[1]],
            z=[center_vec[2]],
            mode="markers",
            marker=dict(
                size=10,
                symbol="x",
                color="red",
                line=dict(width=3, color="black"),
            ),
            name="centers"
        )
        traces.append(centers_trace)

    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True
    )
    fig.update_layout(
            scene_camera=dict(
                eye=dict(x=-2.3, y=1.0, z=2)  # фиксируем положение камеры
            )
        )
    
    fig.show()
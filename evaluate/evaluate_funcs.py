from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import hausdorff
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev
from fastdtw import fastdtw
from pyproj import Geod
from scipy.stats import entropy
from geopy import distance
import math
import numpy as np


def cal_polygon_area(polygon, mode=1):
    """
    计算经纬度多边形的覆盖面积（平方米不会算，先用平方度来做）

    Args:
        polygon (list): 多边形顶点经纬度数组
        mode (int): 1: 平方度， 2：平方千米

    Returns:
        area (float)
    """
    if mode == 1:
        if len(polygon) < 3:
            return 0
        area = Polygon(polygon)
        return area.area
    else:
        if len(polygon) < 3:
            return 0
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(orient(Polygon(polygon)))  # 单位平方米
        return area / 1000000


def arr_to_distribution(arr, min, max, bins=10000):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution


def get_geogradius(rid_lat, rid_lon):
    """
    get the std of the distances of all points away from center as `gyration radius`
    :param trajs:
    :return:
    """
    if len(rid_lat) == 0:
        return 0
    lng1, lat1 = np.mean(rid_lon), np.mean(rid_lat)
    rad = []
    for i in range(len(rid_lat)):
        lng2 = rid_lon[i]
        lat2 = rid_lat[i]
        dis = distance.distance((lat1, lng1), (lat2, lng2)).kilometers
        rad.append(dis)
    rad = np.mean(rad)
    return rad


def js_divergence(p, q):
    """JS散度

    Args:
        p(np.array):
        q(np.array):

    Returns:

    """
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def edit_distance(trace1, trace2):
    """
    the edit distance between two trajectory
    Args:
        trace1:
        trace2:
    Returns:
        edit_distance
    """
    matrix = [[i + j for j in range(len(trace2) + 1)] for i in range(len(trace1) + 1)]
    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            if trace1[i - 1] == trace2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(trace1)][len(trace2)]


def hausdorff_metric(truth, pred, distance='haversine'):
    """豪斯多夫距离
    ref: https://github.com/mavillan/py-hausdorff

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    return hausdorff.hausdorff_distance(truth, pred, distance=distance)


def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = (pow(math.sin(dlat/2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon/2.0), 2.0))
    return R * 2 * math.asin(math.sqrt(a))


def dtw_metric(truth, pred, distance='haversine'):
    """动态时间规整算法
    ref: https://github.com/slaypni/fastdtw

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    if distance == 'haversine':
        distance, path = fastdtw(truth, pred, dist=haversine)
    elif distance == 'manhattan':
        distance, path = fastdtw(truth, pred, dist=cityblock)
    elif distance == 'euclidean':
        distance, path = fastdtw(truth, pred, dist=euclidean)
    elif distance == 'chebyshev':
        distance, path = fastdtw(truth, pred, dist=chebyshev)
    elif distance == 'cosine':
        distance, path = fastdtw(truth, pred, dist=cosine)
    else:
        distance, path = fastdtw(truth, pred, dist=euclidean)
    return distance


rad = math.pi / 180.0
R = 6378137.0


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)

    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the se*cond point
    param lon2: float, longitude of the second point

    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def s_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    # C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def rid_cnt2heat_level(rid_cnt):
    min = 0
    max = np.max(rid_cnt)
    level_num = 100
    bin_size = max // level_num
    rid_heat_level = rid_cnt // bin_size
    return rid_heat_level


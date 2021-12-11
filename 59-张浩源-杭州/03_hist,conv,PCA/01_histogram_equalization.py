import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img_loc = "StaticStorage/lenna.png"


class ImageBasic:
    def __init__(self) -> None:
        pass


class Equilibrium(ImageBasic):
    """
    func: class for array equilibrium
    a_in: one deminsion array (private)
    df: dataframe contain processed info (public)
    """

    def __init__(self, one_d_array):
        super().__init__()
        self.__a_in = one_d_array
        self.df = None

    def __DictGen(self, dict_, v_i):
        """
        func: same like hist, collect var and var-freq
        dict_.key: var name
        dict_.value: var freq
        v_i: var sequence
        """
        dict_[v_i] = 1 if v_i not in dict_.keys() else dict_[v_i] + 1

    def InitDistribution(self):
        """
        func: save hist info to dataframe and sort by var size
        """
        dict_ = {}
        df = pd.DataFrame()
        for i in self.__a_in:
            self.__DictGen(dict_, i)
        df["pix_v"] = dict_.keys()
        df["pix_n"] = dict_.values()
        df = df.sort_values(by="pix_v", ignore_index=True)
        self.df = df

    def MapCalculation(self, v_range):
        """
        func: get the mapping value for each var after equilibrium
        v_range: equilibrium range
        """
        df = self.df
        df["pix_r"] = df.pix_n / len(self.__a_in)
        df["pix_s"] = df.pix_r
        for i in df.index[1:]:
            df.loc[i, "pix_s"] = df.loc[i - 1, "pix_s"] + df.loc[i, "pix_r"]
        df["pix_m"] = df.pix_s * v_range
        df.pix_m = df.pix_m.round().astype(int)
        self.df = df


class ImageProcess(ImageBasic):
    def __init__(self, img_loc):
        super().__init__()
        self.__img = cv2.imread(img_loc)
        self.__h = self.__img.shape[0]
        self.__w = self.__img.shape[1]
        self.__c = self.__img.shape[2]

    @property
    def img(self):
        return self.__img

    def __EmptyImgGen(self, height, width, channel=None, type_=None):
        channel_st = self.__c if not channel else channel
        type_st = self.__img.dtype if not type_ else type_
        img = np.zeros(shape=(height, width, channel_st), dtype=type_st)
        return img

    def HistBalance(self, v_range):
        img = self.__EmptyImgGen(self.__h, self.__w)
        for c in range(self.__c):
            # do a histbalance each channel
            src_c = self.__img[:, :, c]
            dst_c = img[:, :, c]

            hist_c = Equilibrium(src_c.flatten())
            hist_c.InitDistribution()
            hist_c.MapCalculation(v_range)
            c_hist_df = hist_c.df

            for i in c_hist_df.index:
                # use matrix filter and addtion to avoid traversal
                pix_type = c_hist_df.pix_v[i]
                pix_mapv = c_hist_df.pix_m[i]
                dst_c += (src_c == pix_type).astype(np.uint8) * pix_mapv

        return img


def main():
    improcess = ImageProcess(img_loc)
    img_raw = improcess.img
    img_balance = improcess.HistBalance(255)

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.title("origin lenna")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_balance, cv2.COLOR_BGR2RGB))
    plt.title("balance lenna")

    plt.suptitle("Histogram equalization")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

import csv
import requests
import matplotlib.pyplot as plt


def make_csv(dat: str, coord_z: int = 0) -> str:

    lines = dat.splitlines()  

    aerofoil_name = lines.pop(0)
    aerofoil_name = ''.join(e for e in aerofoil_name if e.isalnum())

    with open(aerofoil_name + ".csv", 'w', newline='') as file:
        writer = csv.writer(file) 

        for row in lines:
            writer.writerow([*row.split(), coord_z])

    return aerofoil_name


def get_dat(url: str) -> str:

    r = requests.get(url)
    return r.text


def plot_aerofoil_by_name(name: str) -> None:

    with open(name + ".csv", 'r') as f:
        points = f.read().splitlines()
        x = [float(c.split(",")[0]) for c in points]
        y = [float(c.split(",")[1]) for c in points]
        plt.plot(x, y)
        plt.title(name)
        plt.ylim(-0.5, 0.5)
        plt.xlim(0, 1)
        plt.savefig(name)
        plt.show()


def plot_aerofoil_by_url(url: str) -> None:

    dat = get_dat(url)
    name = make_csv(dat)
    plot_aerofoil_by_name(name)


if __name__ == "__main__":
    url = input("Input the URL of the aerofoil you want on the web.")
    plot_aerofoil_by_url(url)

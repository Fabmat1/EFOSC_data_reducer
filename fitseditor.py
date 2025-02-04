import tkinter as tk
from tkinter import filedialog, messagebox
from astropy.io import fits

class FITSEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("FITS RA/DEC Editor")

        self.filename = None
        self.header = None

        tk.Label(root, text="RA:").grid(row=0, column=0, padx=5, pady=5)
        self.ra_entry = tk.Entry(root, width=30)
        self.ra_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(root, text="DEC:").grid(row=1, column=0, padx=5, pady=5)
        self.dec_entry = tk.Entry(root, width=30)
        self.dec_entry.grid(row=1, column=1, padx=5, pady=5)

        self.load_button = tk.Button(root, text="Load FITS", command=self.load_fits)
        self.load_button.grid(row=2, column=0, padx=5, pady=5)

        self.save_button = tk.Button(root, text="Save FITS", command=self.save_fits)
        self.save_button.grid(row=2, column=1, padx=5, pady=5)

    def load_fits(self):
        self.filename = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits")])
        if not self.filename:
            return

        try:
            with fits.open(self.filename) as hdul:
                self.header = hdul[0].header
                ra = self.header.get("RA", "")
                dec = self.header.get("DEC", "")
                self.ra_entry.delete(0, tk.END)
                self.ra_entry.insert(0, str(ra))
                self.dec_entry.delete(0, tk.END)
                self.dec_entry.insert(0, str(dec))

            messagebox.showinfo("Success", "FITS file loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load FITS file: {e}")

    def save_fits(self):
        if not self.header or not self.filename:
            messagebox.showwarning("Warning", "No FITS file loaded!")
            return

        ra = self.ra_entry.get()
        dec = self.dec_entry.get()

        try:
            with fits.open(self.filename, mode="update") as hdul:
                hdul[0].header["RA"] = ra
                hdul[0].header["DEC"] = dec
                hdul.flush()

            messagebox.showinfo("Success", "FITS file saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save FITS file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FITSEditor(root)
    root.mainloop()

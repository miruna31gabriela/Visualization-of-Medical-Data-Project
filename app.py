import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import vtk
import pydicom
import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import radiomics
from radiomics import featureextractor
import pandas as pd
from matplotlib.dates import DateFormatter
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as msgbox
import logging
logging.getLogger("radiomics").setLevel(logging.WARNING)

DEFAULT_SPACING = [1.0, 1.0, 1.0]
DEFAULT_WINDOW_SIZE = (1600, 800)
DEFAULT_BACKGROUND_COLOR = (0.1, 0.1, 0.1)
TUMOR_THRESHOLD = 50
BREAST_THRESHOLD = 100
PE_THRESHOLD = 0.7
SER_THRESHOLD = 0.9
RADIOMICS_FEATURES = [
    'shape Volume',
    'shape SurfaceArea',
    'shape SurfaceVolumeRatio',
    'shape Sphericity',
    'shape SphericalDisproportion',
    'firstorder Entropy'
]

class TumorVisualizer:
    """A class for visualizing 3D tumor data from DICOM files with interactive features."""

    PLOT_COLORS = [
        '#cc82fa', '#bf1b65', '#f486f7', '#a728c7', '#871435',
        '#007ACC', '#34A853', '#FF6D00', '#00BFA5', '#5D4037'
    ]
    PLOT_MARKERS = ['o', 'x', 's', '^', 'D', 'v', '<', '>', 'P', '*']

    def __init__(self, data_path: str, all_patients: List[str], base_data_path: Path) -> None:
        """Initialize the TumorVisualizer for a specific patient.

        Args:
            data_path: Path to the specific patient's directory (e.g., ".../UCSF-BR-11")
            all_patients: List of all available patient IDs.
            base_data_path: The base path where all patient data is stored.
        """
        self.data_path = Path(data_path)
        self.patient_id = self.data_path.name
        self.all_patients = all_patients
        self.base_data_path = base_data_path
        self.consultations = self.find_segmentation_files(self.data_path)
        self.current_view_mode = "surface"
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}
        self.radiomics_features: Dict[int, Dict[str, float]] = {}
        self.current_consultation = 0
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.extractor.enableFeatureClassByName('shape')
        self.extractor.enableFeatureClassByName('firstorder')

        self.extractor.settings.update({
            'binWidth': 25,
            'interpolator': 'sitkBSpline',
            'resampledPixelSpacing': None,
            'label': 1,
            'normalize': True,
            'normalizeScale': 100,
            'removeOutliers': 3,
            'minimumROIDimensions': 2,
            'minimumROISize': None,
            'geometryTolerance': 1e-6,
            'correctMask': True
        })
        self.consult_mask_folders = {}
        for i, (consult_dir_name, _, _) in enumerate(self.consultations):
            consult_path = self.data_path / consult_dir_name
            
            ser_folder = next(consult_path.glob('*SER*'), None)
            pe_folder = next(consult_path.glob('*PE1*'), None)

            if ser_folder and pe_folder:
                self.consult_mask_folders[i] = [ser_folder, pe_folder]
            else:
                print(f"Warning: Could not find SER or PE1 folders for consultation {consult_dir_name}")

        self._setup_vtk_window()
        self._setup_renderers()
        self._setup_interactor()

        try:
            self._calculate_all_radiomics_features()
        except Exception as e:
            print(f"Warning: Could not calculate radiomics features: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())

    def _setup_vtk_window(self) -> None:
        """Set up the VTK render window."""
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(*DEFAULT_WINDOW_SIZE)
        self.render_window.SetWindowName("Tumor Evolution Visualization")

    def _setup_renderers(self) -> None:
        """Set up renderers for each consultation."""
        self.renderers = []
        num_consults = len(self.consultations)
        if num_consults == 0: return
        viewport_width = 1.0 / num_consults

        for i in range(num_consults):
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(*DEFAULT_BACKGROUND_COLOR)
            renderer.SetViewport(i * viewport_width, 0, (i + 1) * viewport_width, 1)
            self.render_window.AddRenderer(renderer)
            self.renderers.append(renderer)

    def _setup_interactor(self) -> None:
        """Set up the VTK interactor and style."""
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)

    def find_segmentation_files(self, patient_path: Path) -> List[Tuple[str, List[str], List[str]]]:
        """Find segmentation files in the given patient folder path using pathlib.

        Args:
            patient_path: Path to the patient's main folder.

        Returns:
            List of tuples containing (date, breast_files, tumor_files)
        """
        breast_files_by_consult = {}
        tumor_files_by_consult = {}

        if not patient_path.exists():
            print(f"Patient data not found at {patient_path}")
            return []

        segmentation_dirs = [p for p in patient_path.rglob('*') if p.is_dir() and 'Segmentation' in p.name]

        for seg_dir in segmentation_dirs:
            try:
                consult_date = seg_dir.parent.name
                dir_name_lower = seg_dir.name.lower()
                is_tumor = 'pe' in dir_name_lower and 'thresh' in dir_name_lower
                is_breast = 'breast tissue' in dir_name_lower

                dcm_files = sorted([str(f) for f in seg_dir.glob('*.dcm')])
                if not dcm_files:
                    continue

                if is_tumor:
                    tumor_files_by_consult.setdefault(consult_date, []).extend(dcm_files)
                elif is_breast:
                    breast_files_by_consult.setdefault(consult_date, []).extend(dcm_files)
            except IndexError:
                print(f"Warning: Could not determine consultation date for segmentation folder: {seg_dir}")
                continue
        
        for consult in breast_files_by_consult:
            breast_files_by_consult[consult].sort()
        for consult in tumor_files_by_consult:
            tumor_files_by_consult[consult].sort()

        sorted_dates = sorted(breast_files_by_consult.keys() | tumor_files_by_consult.keys())
        
        return [(d, breast_files_by_consult.get(d, []), tumor_files_by_consult.get(d, [])) 
                for d in sorted_dates]

    def get_cached_data(self, consult_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Get cached data for a consultation or load it if not cached.

        Args:
            consult_idx: Index of the consultation

        Returns:
            Tuple of (breast_data, tumor_data, date)
        """
        if consult_idx in self.cache:
            return self.cache[consult_idx]

        date, breast_files, tumor_files = self.consultations[consult_idx]
        breast_data = None
        tumor_data = None
        
        # Process tumor files first
        if tumor_files:
            all_slices = []
            for file_path in tumor_files:
                ds = pydicom.dcmread(file_path)
                pixel_data = ds.pixel_array
                all_slices.append(pixel_data)
            if all_slices:
                tumor_data = np.stack(all_slices, axis=0)
                tumor_data = np.squeeze(tumor_data)
                if tumor_data.ndim != 3:
                    print(f"ERROR: tumor_data shape is {tumor_data.shape}, expected 3D")
        if breast_files:
            all_slices = []
            for file_path in breast_files:
                ds = pydicom.dcmread(file_path)
                pixel_data = ds.pixel_array
                all_slices.append(pixel_data)
            if all_slices:
                breast_data = np.stack(all_slices, axis=0)
                breast_data = np.squeeze(breast_data)
                if breast_data.ndim != 3:
                    print(f"ERROR: breast_data shape is {breast_data.shape}, expected 3D")

        if tumor_data is not None and breast_data is not None and tumor_data.shape != breast_data.shape:
            if tumor_data.shape[0] < breast_data.shape[0]:
                pad_width = ((0, breast_data.shape[0] - tumor_data.shape[0]), (0, 0), (0, 0))
                tumor_data = np.pad(tumor_data, pad_width, mode='constant', constant_values=0)
            else:
                tumor_data = tumor_data[:breast_data.shape[0], :, :]

        self.cache[consult_idx] = (breast_data, tumor_data, date)
        return breast_data, tumor_data, date

    def numpy_to_vtk_image(self, arr: np.ndarray) -> vtk.vtkImageData:
        """Convert numpy array to VTK image data.

        Args:
            arr: 3D numpy array (z, y, x)

        Returns:
            VTK image data object
        """
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
        vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        flat = arr.ravel(order='C')
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_img.GetPointData().SetScalars(vtk_array)
        return vtk_img

    def create_actors(self, breast_data: Optional[np.ndarray],
                     tumor_data: Optional[np.ndarray],
                     view_mode: str = "surface") -> Tuple[Optional[vtk.vtkActor],
                                                        Optional[vtk.vtkActor],
                                                        Optional[vtk.vtkVolume],
                                                        Optional[vtk.vtkVolume]]:
        """Create VTK actors for visualization.

        Args:
            breast_data: 3D numpy array of breast data
            tumor_data: 3D numpy array of tumor data
            view_mode: Visualization mode ("surface", "volume", "tumor_only", "breast_only")

        Returns:
            Tuple of (breast_actor, tumor_actor, breast_volume, tumor_volume)
        """
        breast_actor, tumor_actor = None, None
        breast_volume, tumor_volume = None, None
        breast_vtk = self.numpy_to_vtk_image(breast_data) if breast_data is not None else None
        tumor_vtk = self.numpy_to_vtk_image(tumor_data) if tumor_data is not None else None

        if view_mode == "surface":
            breast_actor, tumor_actor = self._create_surface_actors(breast_vtk, tumor_vtk)
        elif view_mode == "volume":
            breast_volume, tumor_volume = self._create_volume_actors(breast_vtk, tumor_vtk)
        elif view_mode == "tumor_only":
            _, tumor_actor = self._create_surface_actors(None, tumor_vtk)
        elif view_mode == "breast_only":
            breast_actor, _ = self._create_surface_actors(breast_vtk, None)

        return breast_actor, tumor_actor, breast_volume, tumor_volume

    def _create_surface_actors(self, breast_vtk: Optional[vtk.vtkImageData],
                             tumor_vtk: Optional[vtk.vtkImageData]) -> Tuple[Optional[vtk.vtkActor],
                                                                           Optional[vtk.vtkActor]]:
        """Create surface actors for breast and tumor data."""
        breast_actor, tumor_actor = None, None

        if breast_vtk is not None:
            breast_actor = self._create_surface_actor(breast_vtk, BREAST_THRESHOLD, (0.0, 0.0, 1.0), 0.1)

        if tumor_vtk is not None:
            tumor_actor = self._create_surface_actor(tumor_vtk, TUMOR_THRESHOLD, (1.0, 0.0, 0.0), 1.0)

        return breast_actor, tumor_actor

    def _create_surface_actor(self, vtk_data: vtk.vtkImageData, threshold: float,
                            color: Tuple[float, float, float], opacity: float) -> vtk.vtkActor:
        """Create a single surface actor with given parameters."""
        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(vtk_data)
        surface.SetValue(0, threshold)

        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(surface.GetOutputPort())
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(connectivity.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)

        if opacity == 1.0:  
            actor.GetProperty().SetAmbient(0.4)
            actor.GetProperty().SetDiffuse(0.9)
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(10)
            actor.GetProperty().SetRepresentationToSurface()

        return actor

    def _create_volume_actors(self, breast_vtk: Optional[vtk.vtkImageData],
                            tumor_vtk: Optional[vtk.vtkImageData]) -> Tuple[Optional[vtk.vtkVolume],
                                                                          Optional[vtk.vtkVolume]]:
        """Create volume actors for breast and tumor data."""
        breast_volume, tumor_volume = None, None

        if breast_vtk is not None:
            breast_volume = self._create_volume_actor(breast_vtk, (0.0, 0.0, 1.0), 0.1)

        if tumor_vtk is not None:
            tumor_volume = self._create_volume_actor(tumor_vtk, (1.0, 0.0, 0.0), 1.0)

        return breast_volume, tumor_volume

    def _create_volume_actor(self, vtk_data: vtk.vtkImageData, color: Tuple[float, float, float],
                           opacity: float) -> vtk.vtkVolume:
        """Create a single volume actor with given parameters."""
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_data)

        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)

        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(255, *color)

        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(0, 0.0)
        opacity_tf.AddPoint(255, opacity)

        property = vtk.vtkVolumeProperty()
        property.SetColor(color_tf)
        property.SetScalarOpacity(opacity_tf)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()

        volume.SetProperty(property)
        return volume

    def update_visualization(self) -> None:
        """Update the visualization with current data and settings."""
        for i, renderer in enumerate(self.renderers):
            renderer.RemoveAllViewProps()
            breast_data, tumor_data, date = self.get_cached_data(i)
            breast_actor, tumor_actor, breast_volume, tumor_volume = self.create_actors(
                breast_data, tumor_data, self.current_view_mode)

            if self.current_view_mode == "surface":
                if breast_actor:
                    renderer.AddActor(breast_actor)
                if tumor_actor:
                    renderer.AddActor(tumor_actor)
            elif self.current_view_mode == "volume":
                if breast_volume:
                    renderer.AddVolume(breast_volume)
                if tumor_volume:
                    renderer.AddVolume(tumor_volume)
            elif self.current_view_mode == "tumor_only":
                if tumor_actor:
                    renderer.AddActor(tumor_actor)
            elif self.current_view_mode == "breast_only":
                if breast_actor:
                    renderer.AddActor(breast_actor)

            self._add_text_overlays(renderer, i, date)
            renderer.ResetCamera()

        self.render_window.Render()

    def _add_text_overlays(self, renderer: vtk.vtkRenderer, consult_idx: int, date: str) -> None:
        """Add text overlays to the renderer."""
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Consult {consult_idx+1}\n{date}\nMode: {self.current_view_mode}")
        text_actor.GetTextProperty().SetFontSize(14)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.SetPosition(0.05, 0.05)
        renderer.AddActor2D(text_actor)

        tumor_volume_text = "Tumor volume: N/A"
        if consult_idx in self.radiomics_features:
            tumor_volume_mm3 = self.radiomics_features[consult_idx].get('original_shape_MeshVolume', 0)
            if tumor_volume_mm3 > 0:
                tumor_volume_text = f"Tumor volume: {tumor_volume_mm3:.2f} mm³"
            else:
                tumor_volume_text = "Tumor volume: N/A"

        volume_actor = vtk.vtkTextActor()
        volume_actor.SetInput(tumor_volume_text)
        volume_actor.GetTextProperty().SetFontSize(14)
        volume_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        volume_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        volume_actor.SetPosition(0.05, 0.90)
        renderer.AddActor2D(volume_actor)

    def print_radiomics_features_gui(self):
        if not self.radiomics_features:
            msgbox.showinfo("Radiomics Features", "No radiomics features available.")
            return

        rows = []
        for i in sorted(self.radiomics_features.keys()):
            date_str = '-'.join(self.consultations[i][0].split('-')[:3])
            row = self.radiomics_features[i].copy()
            row['Date'] = date_str
            rows.append(row)
        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date')
        df = df.sort_index()

        suggestive_headers = {
            'original_shape_MeshVolume': 'Tumor Volume (mm³)',
            'original_shape_SurfaceArea': 'Surface Area (mm²)',
            'original_shape_SurfaceVolumeRatio': 'Surface Area to Volume Ratio (mm⁻¹)',
            'original_shape_Sphericity': 'Sphericity',
            'original_shape_SphericalDisproportion': 'Spherical Disproportion',
            'original_firstorder_Entropy': 'Entropy'
        }

        columns_to_display = [key for key in suggestive_headers.keys() if key in df.columns]
        df_display = df[columns_to_display]

        table_win = tk.Toplevel(self.menu_window)
        table_win.title("Radiomics Features Table")
        table_win.geometry("1150x250")

        tree_columns = [suggestive_headers[key] for key in columns_to_display]
        tree = ttk.Treeview(table_win, columns=['Date'] + tree_columns, show='headings')
        tree.pack(fill=tk.BOTH, expand=True)

        tree.heading('Date', text='Date')
        tree.column('Date', width=140)
        for key in columns_to_display:
            header_text = suggestive_headers[key]
            tree.heading(header_text, text=header_text)
            tree.column(header_text, width=150, anchor='center')

        for idx, row in df_display.iterrows():
            date_formatted = idx.strftime('%Y-%m-%d')
            values = [date_formatted] + [f"{row[key]:.6f}" for key in columns_to_display]
            tree.insert('', 'end', values=values)

        scrollbar = ttk.Scrollbar(table_win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    def plot_time_series_features_gui(self):
        try:
            plt.ion()
            self.plot_time_series_features(block=True)
        except Exception as e:
            msgbox.showerror("Error", str(e))

    def _build_radiomics_df(self) -> pd.DataFrame:
        if not self.radiomics_features:
            return pd.DataFrame()

        rows = []
        for i in sorted(self.radiomics_features.keys()):
            if i >= len(self.consultations):
                print(f"Warning: Radiomics data found for index {i}, but no corresponding consultation found.")
                continue
            date_str = '-'.join(self.consultations[i][0].split('-')[:3])
            row = self.radiomics_features[i].copy()
            row['Date'] = date_str
            row['Consultation'] = i 
            rows.append(row)
        
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date')
        df = df.set_index('Consultation')

        df = df.rename(columns={
            'original_shape_MeshVolume': 'shape Volume',
            'original_shape_SurfaceArea': 'shape SurfaceArea',
            'original_shape_SurfaceVolumeRatio': 'shape SurfaceVolumeRatio',
            'original_shape_Sphericity': 'shape Sphericity',
            'original_shape_SphericalDisproportion': 'shape SphericalDisproportion',
            'original_firstorder_Entropy': 'firstorder Entropy'
        })
        return df

    def plot_time_series_features(self, block=False):
        if not self.radiomics_features:
            msgbox.showinfo("Info", "No radiomics features available.")
            return

        df = self._build_radiomics_df()
        if df.empty:
            msgbox.showinfo("Info", "Could not process radiomics data for plotting.")
            return

        feature_titles = {
            'shape Volume': 'Tumor Volume (mm³)',
            'shape SurfaceArea': 'Surface Area (mm²)',
            'shape SurfaceVolumeRatio': 'Surface Area to Volume Ratio (mm⁻¹)',
            'shape Sphericity': 'Sphericity',
            'shape SphericalDisproportion': 'Spherical Disproportion',
            'firstorder Entropy': 'Entropy'
        }
        ylabels = {
            'shape Volume': 'Volume (mm³)',
            'shape SurfaceArea': 'Surface Area (mm²)',
            'shape SurfaceVolumeRatio': 'A/V (mm⁻¹)',
            'shape Sphericity': 'Sphericity',
            'shape SphericalDisproportion': 'Spherical Disproportion',
            'firstorder Entropy': 'Entropy'
        }
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        dfs = {self.patient_id: df}

        for idx, feature in enumerate(RADIOMICS_FEATURES):
            self._configure_plot_axes(axes[idx], feature, feature_titles, ylabels, dfs)
        for idx in range(len(RADIOMICS_FEATURES), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'Tumor Radiomics Features for {self.patient_id}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if block:
            plt.show(block=False)
        else:
            plt.show()

    def _configure_plot_axes(self, ax: plt.Axes, feature: str, feature_titles: Dict, ylabels: Dict, dfs: Dict[str, pd.DataFrame]):
        """Helper to configure a single subplot for radiomics plotting."""
        ax.set_title(feature_titles.get(feature, feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabels.get(feature, feature), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='y', labelsize=9)

        has_data_for_feature = False
        patient_ids = list(dfs.keys())

        for i, patient_id in enumerate(patient_ids):
            df = dfs[patient_id]
            if feature in df.columns:
                has_data_for_feature = True
                color = self.PLOT_COLORS[i % len(self.PLOT_COLORS)]
                marker = self.PLOT_MARKERS[i % len(self.PLOT_MARKERS)]
                ax.plot(df.index, df[feature], marker=marker, linestyle='-', label=patient_id, markersize=5, color=color)

        all_indices = [idx for df in dfs.values() for idx in df.index]

        if all_indices:
            max_index = max(all_indices) if all_indices else 0
            ticks = range(max_index + 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"Consult {i+1}" for i in ticks], rotation=30, ha='right')
            ax.tick_params(axis='x', labelsize=9)
        if ax.get_legend_handles_labels()[1]:
            ax.legend()

        if not has_data_for_feature:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

    def start(self) -> None:
        if not self.renderers:
            msgbox.showerror("Error", "No data loaded or no consultations found for this patient.")
            return
        self.menu_window.withdraw()
        self.update_visualization()
        self.interactor.Initialize()
        self.interactor.Start()
        self.menu_window.deiconify()

    def _show_multi_patient_selector(self):
        selector_window = tk.Toplevel(self.menu_window)
        selector_window.title("Select Comparison Patients")
        selector_window.geometry("400x300")

        frame = ttk.Frame(selector_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        label = ttk.Label(frame, text="Select patient(s) to compare with (use Ctrl+Click):", font=('Arial', 12))
        label.pack(pady=5)

        comparison_patients = [p for p in self.all_patients if p != self.patient_id]
        if not comparison_patients:
            msgbox.showinfo("Info", "No other patients available to compare.")
            selector_window.destroy()
            return

        listbox_frame = ttk.Frame(frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, font=('Arial', 11))
        for patient in comparison_patients:
            listbox.insert(tk.END, patient)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill='y')

        def on_select():
            selected_indices = listbox.curselection()
            if not selected_indices:
                msgbox.showwarning("Warning", "Please select at least one patient.")
                return

            selected_patient_ids = [listbox.get(i) for i in selected_indices]
            
            selector_window.destroy()
            self._launch_multi_patient_comparison_plot(selected_patient_ids)
        
        select_btn = ttk.Button(frame, text="Compare", command=on_select)
        select_btn.pack(pady=10)

    def _launch_multi_patient_comparison_plot(self, comparison_patient_ids: List[str]):
        #Loads data for multiple comparison patients and launches the comparison plot.
        try:
            root = tk.Toplevel()
            root.withdraw()
            msgbox.showinfo("Loading", f"Loading data for {len(comparison_patient_ids)} comparison patient(s).\nPlease wait...", parent=root)
            root.destroy()

            comparison_visualizers = []
            for patient_id in comparison_patient_ids:
                comparison_patient_path = self.base_data_path / patient_id
                visualizer = TumorVisualizer(
                    str(comparison_patient_path), 
                    self.all_patients, 
                    self.base_data_path
                )
                if not visualizer.radiomics_features:
                    msgbox.showwarning("Warning", f"No radiomics features found for patient {patient_id}. They will be skipped.")
                else:
                    comparison_visualizers.append(visualizer)

            if not self.radiomics_features:
                msgbox.showwarning("Warning", f"No radiomics features available for the primary patient ({self.patient_id}). Cannot perform comparison.")
                return

            if not comparison_visualizers:
                msgbox.showwarning("Warning", "No valid data could be loaded for any of the selected comparison patients.")
                return

            self.plot_multi_patient_time_series(comparison_visualizers)

        except Exception as e:
            msgbox.showerror("Error", f"Failed to load or plot comparison data: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_multi_patient_time_series(self, comparison_visualizers: List["TumorVisualizer"]):
        """
        Plots a time series comparison of radiomics features for the primary patient
        and a list of comparison patients.
        """
        all_visualizers = [self] + comparison_visualizers
        all_dfs = {viz.patient_id: viz._build_radiomics_df() for viz in all_visualizers}

        all_dfs = {pid: df for pid, df in all_dfs.items() if not df.empty}

        if not all_dfs:
            msgbox.showinfo("Info", "No patients have radiomics data to plot.")
            return
        
        feature_titles = {
            'shape Volume': 'Tumor Volume (mm³)',
            'shape SurfaceArea': 'Surface Area (mm²)',
            'shape SurfaceVolumeRatio': 'Surface Area to Volume Ratio (mm⁻¹)',
            'shape Sphericity': 'Sphericity',
            'shape SphericalDisproportion': 'Spherical Disproportion',
            'firstorder Entropy': 'Entropy'
        }
        ylabels = {
            'shape Volume': 'Volume (mm³)',
            'shape SurfaceArea': 'Surface Area (mm²)',
            'shape SurfaceVolumeRatio': 'A/V (mm⁻¹)',
            'shape Sphericity': 'Sphericity',
            'shape SphericalDisproportion': 'Spherical Disproportion',
            'firstorder Entropy': 'Entropy'
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(RADIOMICS_FEATURES):
            self._configure_plot_axes(axes[idx], feature, feature_titles, ylabels, all_dfs)

        for idx in range(len(RADIOMICS_FEATURES), len(axes)):
            axes[idx].set_visible(False)

        title_patients = ", ".join(all_dfs.keys())
        fig.suptitle(f'Radiomics Comparison: {title_patients}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    def print_pe_ser_tumor_volumes(self, consult_idx: int) -> None:
        try:
            pe_folder = self.consult_mask_folders[consult_idx][1]
            ser_folder = self.consult_mask_folders[consult_idx][0]
            pe_files = sorted([os.path.join(pe_folder, f) for f in os.listdir(pe_folder) if f.endswith('.dcm')])
            if not pe_files:
                msgbox.showinfo("Info", f"No DICOM files found in PE folder: {pe_folder}")
                return
            pe_masks = self._process_dicom_slices(pe_files, PE_THRESHOLD)
            ser_files = sorted([os.path.join(ser_folder, f) for f in os.listdir(ser_folder) if f.endswith('.dcm')])
            if not ser_files:
                msgbox.showinfo("Info", f"No DICOM files found in SER folder: {ser_folder}")
                return
            ser_masks = self._process_dicom_slices(ser_files, SER_THRESHOLD)
            spacing = self._get_spacing_from_dicom(pe_folder)
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            total_pe_voxels = sum(np.sum(mask) for mask in pe_masks)
            total_ser_voxels = sum(np.sum(mask) for mask in ser_masks)
            total_pe_volume = total_pe_voxels * voxel_volume
            total_ser_volume = total_ser_voxels * voxel_volume 
            msg = (f"Results for Consultation {consult_idx + 1}:\n"
                   f"PE mask total voxels: {total_pe_voxels}\n"
                   f"PE mask total volume: {total_pe_volume/1000:.2f} mm³\n"
                   f"SER mask total voxels: {total_ser_voxels}\n"
                   f"SER mask total volume: {total_ser_volume/1000:.2f} mm³")
            msgbox.showinfo("Tumor Volumes", msg)
        except Exception as e:
            msgbox.showerror("Error", str(e))

    def _process_dicom_slices(self, files: List[str], threshold: float) -> List[np.ndarray]:
        """Process DICOM slices and create masks."""
        masks = []
        for f in files:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
            arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            mask = (arr_norm > threshold)
            masks.append(mask)
        return masks

    def _get_spacing_from_dicom(self, folder_path: Path) -> List[float]:
        try:
            dcm_files = list(folder_path.glob('*.dcm'))
            if not dcm_files:
                print(f"No DICOM files found in {folder_path}")
                return DEFAULT_SPACING

            ds = pydicom.dcmread(str(dcm_files[0]))

            if hasattr(ds, 'PixelSpacing') and hasattr(ds, 'SliceThickness'):
                return [float(ds.PixelSpacing[0]),
                       float(ds.PixelSpacing[1]),
                       float(ds.SliceThickness)]
            return DEFAULT_SPACING

        except Exception as e:
            print(f"Error reading DICOM file in {folder_path}: {str(e)}")
            return DEFAULT_SPACING

    def _load_dicom_series_from_folder(self, folder_path: Path) -> Optional[np.ndarray]:
        """Loads a DICOM series from a folder into a 3D numpy array."""
        if not folder_path or not folder_path.exists():
            print(f"Warning: DICOM folder path does not exist: {folder_path}")
            return None
        
        dicom_files = sorted([f for f in folder_path.glob('*.dcm')])
        if not dicom_files:
            print(f"Warning: No DICOM files found in {folder_path}")
            return None
        
        all_slices = []
        for file_path in dicom_files:
            ds = pydicom.dcmread(file_path)
            pixel_data = ds.pixel_array
            all_slices.append(pixel_data)
        
        if not all_slices:
            return None
            
        image_data = np.stack(all_slices, axis=0)
        image_data = np.squeeze(image_data)
        
        if image_data.ndim != 3:
            print(f"Warning: Loaded image data from {folder_path} is not 3D. Shape is {image_data.shape}")
            return None
            
        return image_data

    def _calculate_all_radiomics_features(self) -> None:
        for i in range(len(self.consultations)):
            self._calculate_radiomics_features(i)

    def _calculate_radiomics_features(self, consult_idx: int) -> None:
        """
        Calculate radiomics features using pyradiomics, with a manual calculation
        for Spherical Disproportion to ensure its availability.
        The actual image data (from PE1) is used for first-order feature calculation,
        while the tumor segmentation is used as the mask.
        """
        try:
            _, tumor_data, _ = self.get_cached_data(consult_idx)
            if tumor_data is None:
                print(f"Warning: No tumor segmentation data for consultation {consult_idx + 1}, skipping radiomics.")
                return

            if consult_idx not in self.consult_mask_folders:
                print(f"Warning: No mask folders found for consultation {consult_idx + 1}, skipping radiomics.")
                return

            image_folder = self.consult_mask_folders[consult_idx][1]  
            image_data = self._load_dicom_series_from_folder(image_folder)

            if image_data is None:
                print(f"Warning: Could not load image data from {image_folder} for consultation {consult_idx + 1}.")
                return
            if image_data.shape != tumor_data.shape:
                print(f"Warning: Image ({image_data.shape}) and mask ({tumor_data.shape}) shapes do not match for consultation {consult_idx + 1}. Attempting to align.")
                
                if image_data.shape[0] > tumor_data.shape[0]: 
                    pad_width = ((0, image_data.shape[0] - tumor_data.shape[0]), (0, 0), (0, 0))
                    tumor_data = np.pad(tumor_data, pad_width, mode='constant', constant_values=0)
                else: 
                    tumor_data = tumor_data[:image_data.shape[0], :, :]

                if image_data.shape != tumor_data.shape:
                    print(f"Error: Could not align image and mask shapes for consultation {consult_idx + 1} after adjustment. Aborting radiomics.")
                    return
                else:
                    print(f"Successfully aligned shapes to {image_data.shape}.")

            mask = tumor_data > 0
            if np.sum(mask) == 0:
                print(f"Warning: Empty mask generated for consultation {consult_idx + 1}")
                return

            spacing = self._get_spacing_from_dicom(image_folder)
            import SimpleITK as sitk
            
            image_sitk = sitk.GetImageFromArray(image_data.astype(np.float32))
            image_sitk.SetSpacing(spacing)
            
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_sitk.SetSpacing(spacing)
            
            features = self.extractor.execute(image_sitk, mask_sitk)
            if not features:
                print(f"Warning: pyradiomics returned no features for consultation {consult_idx + 1}")
                return

            features_to_store = {key: float(val) for key, val in features.items() if key.startswith('original_')}
            if 'original_shape_MeshVolume' in features_to_store:
                features_to_store['original_shape_MeshVolume'] /= 1000.0
            if 'original_shape_SurfaceArea' in features_to_store:
                features_to_store['original_shape_SurfaceArea'] /= 1000.0
            self.radiomics_features[consult_idx] = features_to_store
            
            sphericity = self.radiomics_features[consult_idx].get('original_shape_Sphericity')
            if sphericity is not None and sphericity != 0:
                self.radiomics_features[consult_idx]['original_shape_SphericalDisproportion'] = 1 / sphericity
            elif 'original_shape_SphericalDisproportion' not in self.radiomics_features[consult_idx]:
                self.radiomics_features[consult_idx]['original_shape_SphericalDisproportion'] = 0.0

            print(f"Successfully calculated radiomics for consultation {consult_idx + 1}.")

        except Exception as e:
            print(f"Error calculating radiomics features for consultation {consult_idx + 1}: {e}")
            import traceback
            traceback.print_exc()

    def show_menu(self) -> None:
        """Display the main menu window."""
        self.menu_window = tk.Tk()
        self.menu_window.title("Tumor Visualization Menu")
        self.menu_window.geometry("500x650")

        style = ttk.Style()
        style.configure("Menu.TButton", padding=10, font=('Arial', 12))
        main_frame = ttk.Frame(self.menu_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        title_label = ttk.Label(main_frame, text="Tumor Visualization Menu", font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)

        vis_label = ttk.Label(main_frame, text="Visualization Modes", font=('Arial', 13, 'bold'))
        vis_label.pack(pady=(10, 5))
        vis_buttons = [
            ("Launch 3D Viewer", self.start), 
            ("Surface View", lambda: self._change_view_mode_gui("surface")),
            ("Volume View", lambda: self._change_view_mode_gui("volume")),
            ("Tumor Only View", lambda: self._change_view_mode_gui("tumor_only")),
            ("Breast Only View", lambda: self._change_view_mode_gui("breast_only"))
        ]
        for text, command in vis_buttons:
            btn = ttk.Button(main_frame, text=text, command=command, style="Menu.TButton")
            btn.pack(fill=tk.X, pady=3)

        analysis_label = ttk.Label(main_frame, text="Tumor Analysis", font=('Arial', 13, 'bold'))
        analysis_label.pack(pady=(20, 5))
        analysis_buttons = [
            ("Show Radiomics Features", self.print_radiomics_features_gui),
            ("Plot Time Series Features", self.plot_time_series_features_gui),
            ("Compare Radiomics with Other Patients", self._show_multi_patient_selector)
        ]
        for text, command in analysis_buttons:
            btn = ttk.Button(main_frame, text=text, command=command, style="Menu.TButton")
            btn.pack(fill=tk.X, pady=3)
        btn = ttk.Button(main_frame, text="Exit", command=self.menu_window.destroy, style="Menu.TButton")
        btn.pack(fill=tk.X, pady=(30, 3))
        self.menu_window.mainloop()

    def _change_view_mode_gui(self, mode):
        self.current_view_mode = mode
        try:
            self.update_visualization()
        except Exception as e:
            import traceback
            msgbox.showerror("Rendering Error", f"Failed to switch to {mode} view.\n\n{str(e)}\n\n{traceback.format_exc()}")

def main():
    """Launches the patient selection GUI and then the main application."""
    base_data_path = Path("TUWIEN/VisMedData")

    try:
        patients = sorted([p.name for p in base_data_path.iterdir() if p.is_dir() and p.name.startswith('UCSF-BR-')])
    except FileNotFoundError:
        msgbox.showerror("Error", f"Base data path not found: {base_data_path}")
        return

    if not patients:
        msgbox.showerror("Error", f"No patient folders (UCSF-BR-*) found in {base_data_path}")
        return

    selector_window = tk.Tk()
    selector_window.title("Select Patient")
    selector_window.geometry("350x150")
    
    frame = ttk.Frame(selector_window, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)

    label = ttk.Label(frame, text="Please select a patient to load:", font=('Arial', 12))
    label.pack(pady=5)

    patient_var = tk.StringVar()
    patient_combo = ttk.Combobox(frame, textvariable=patient_var, values=patients, state="readonly", font=('Arial', 11))
    if patients:
        patient_combo.current(0)
    patient_combo.pack(pady=10, fill=tk.X)

    def load_patient():
        selected_patient_id = patient_var.get()
        if not selected_patient_id:
            msgbox.showwarning("Warning", "Please select a patient.")
            return
            
        patient_path = base_data_path / selected_patient_id
        selector_window.destroy() 
        
        visualizer = TumorVisualizer(str(patient_path), patients, base_data_path)
        visualizer.show_menu()

    load_btn = ttk.Button(frame, text="Load Patient", command=load_patient)
    load_btn.pack(pady=10)

    selector_window.mainloop()


if __name__ == "__main__":
    main()

class MicroscopeManager:
    _instance = None
    _microscope = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MicroscopeManager()
        return cls._instance

    def set_microscope(self, microscope):
        self._microscope = microscope
        return "microscope_1"  # Return a reference ID

    def get_microscope(self, microscope_id):
        if microscope_id == "microscope_1" and self._microscope is not None:
            return self._microscope
        raise ValueError(f"No microscope found for ID: {microscope_id}")

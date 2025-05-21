from activities.readFile import ReadFileActivity
from activities.transform import TransformActivity
from activities.loadFile import LoadFileActivity
from activities.demo_transformation import TransformationActivity

class ActivityFactory:
    _mapping = {
        'file_input': ReadFileActivity,
        'transformation': TransformActivity,
        'file_output': LoadFileActivity,
        'transform': TransformationActivity
    }

    @staticmethod
    def create(activity_def: dict):
        typ = activity_def.get('type')
        cls = ActivityFactory._mapping.get(typ)
        if not cls:
            raise ValueError(f"Unknown activity type: {typ}")
        return cls(activity_def)

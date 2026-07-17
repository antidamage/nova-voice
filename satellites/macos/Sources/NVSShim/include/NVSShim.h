#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Runs the block, converting any raised Objective-C exception into an
/// NSError.  AVAudioEngine graph mutations (connect, start, player start)
/// raise NSExceptions when the output device is mid-reconfiguration; Swift
/// cannot catch those, and an uncaught one terminates the whole satellite —
/// including microphone capture.
BOOL NVSCatchException(void (NS_NOESCAPE ^block)(void),
                       NSError *_Nullable *_Nullable error);

NS_ASSUME_NONNULL_END

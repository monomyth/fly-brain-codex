import tarfile
from fly_brain.publication import portable, anonymous_tar_member


def test_provenance_paths_are_portable_and_urls_survive():
    item={'datasets':['/Users/researcher/code/codex/fly-brain/data/example'],
          'reference':'/home/researcher/code/data/malecns/v1.0',
          'other':'/Users/researcher/Documents/example',
          'source':'https://github.com/example-account/fly-brain-codex',
          'host':'private-workstation', 'value':0.75}
    assert portable(item)=={'datasets':['project:/data/example'],'reference':'reference:/v1.0',
                            'other':'~/Documents/example','source':item['source'],'host':'gpu-server','value':.75}


def test_private_host_names_are_removed_from_free_text_and_private_urls():
    result=portable({'note':'Run on Workstation-Test; see http://workstation-test.local/run',
                     'source':'https://huggingface.co/example-account/model'},
                    redactions=['workstation-test.local','workstation-test'])
    assert result['note']=='Run on gpu-server; see http://gpu-server/run'
    assert result['source']=='https://huggingface.co/example-account/model'


def test_archive_ownership_and_extended_metadata_are_anonymous():
    member=tarfile.TarInfo('model.npz');member.uid=501;member.gid=20
    member.uname='researcher';member.gname='staff';member.mtime=123456
    member.pax_headers={'atime':'123','SCHILY.dev':'42'}
    cleaned=anonymous_tar_member(member)
    assert (cleaned.uid,cleaned.gid,cleaned.uname,cleaned.gname,cleaned.mtime)==(0,0,'','',0)
    assert cleaned.pax_headers=={}
    assert cleaned.name=='model.npz'


def test_archive_symlinks_cannot_disclose_private_targets():
    import pytest
    member=tarfile.TarInfo('alias');member.type=tarfile.SYMTYPE
    member.linkname='/Users/researcher/private-file'
    with pytest.raises(ValueError,match='regular files'):
        anonymous_tar_member(member)
